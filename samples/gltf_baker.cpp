/*
 * Copyright (C) 2019 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define GLTFIO_SIMPLEVIEWER_IMPLEMENTATION
#define DEBUG_PATHTRACER 0

#include "app/Config.h"
#include "app/FilamentApp.h"
#include "app/IBL.h"

#include <filament/Engine.h>
#include <filament/Scene.h>
#include <filament/View.h>

#include <gltfio/AssetLoader.h>
#include <gltfio/AssetPipeline.h>
#include <gltfio/FilamentAsset.h>
#include <gltfio/ResourceLoader.h>
#include <gltfio/SimpleViewer.h>

#include <image/ImageOps.h>
#include <image/LinearImage.h>

#include <imageio/ImageEncoder.h>

#include <utils/NameComponentManager.h>
#include <utils/JobSystem.h>

#include <math/vec2.h>

#include <getopt/getopt.h>

#include <atomic>
#include <fstream>
#include <string>

#include "generated/resources/gltf.h"
#include "generated/resources/resources.h"

using namespace filament;
using namespace gltfio;
using namespace utils;

static inline ImVec2 operator+(const ImVec2& lhs, const ImVec2& rhs) {
    return { lhs.x+rhs.x, lhs.y+rhs.y };
}

enum AppState {
    EMPTY,
    LOADED,
    RENDERING,
    PARAMETRIZING,
    PARAMETRIZED,
    BAKING,
    BAKED,
    EXPORTED,
};

enum class ResultsVisualization : int {
    MESH_ORIGINAL,
    MESH_MODIFIED,
    MESH_PREVIEW_AO,
    MESH_PREVIEW_UV,
    IMAGE_OCCLUSION,
    IMAGE_BENT_NORMALS
};

enum ExportOption : int {
    VISUALIZE_AO,
    VISUALIZE_UV,
    PRESERVE_MATERIALS,
};

struct App {
    Engine* engine;
    Camera* camera;
    SimpleViewer* viewer;
    Config config;
    NameComponentManager* names;
    MaterialProvider* materials;
    View* overlayView = nullptr;
    Scene* overlayScene = nullptr;
    VertexBuffer* overlayVb = nullptr;
    IndexBuffer* overlayIb = nullptr;
    Texture* overlayTexture = nullptr;
    MaterialInstance* overlayMaterial = nullptr;
    utils::Entity overlayEntity;

    AssetLoader* loader = nullptr;
    gltfio::AssetPipeline* pipeline = nullptr;

    FilamentAsset* viewerAsset = nullptr;
    gltfio::AssetPipeline::AssetHandle currentAsset = nullptr;
    gltfio::AssetPipeline::AssetHandle modifiedAsset = nullptr;
    gltfio::AssetPipeline::AssetHandle previewAoAsset = nullptr;
    gltfio::AssetPipeline::AssetHandle previewUvAsset = nullptr;

    bool hasTestRender = false;

    image::LinearImage ambientOcclusion;
    image::LinearImage bentNormals;
    image::LinearImage meshNormals;
    image::LinearImage meshPositions;

    bool actualSize = false;
    AppState state = EMPTY;
    utils::Path filename;
    ResultsVisualization resultsVisualization = ResultsVisualization::MESH_ORIGINAL;
    AppState pushedState;
    uint32_t bakeResolution = 1024;
    int samplesPerPixel = 256;
    ExportOption exportOption = PRESERVE_MATERIALS;

    // Secondary threads might write to the following fields.
    std::shared_ptr<std::string> statusText;
    std::shared_ptr<std::string> messageBoxText;
    std::atomic<bool> requestOverlayUpdate;
    std::atomic<bool> requestStatePop;
};

struct OverlayVertex {
    filament::math::float2 position;
    filament::math::float2 uv;
};

static OverlayVertex OVERLAY_VERTICES[4] = {
    {{0, 0}, {0, 0}},
    {{ 1000, 0}, {1, 0}},
    {{0,  1000}, {0, 1}},
    {{ 1000,  1000}, {1, 1}},
};

static const char* DEFAULT_IBL = "envs/venetian_crossroads";

static const char* INI_FILENAME = "gltf_baker.ini";

static void printUsage(char* name) {
    std::string exec_name(Path(name).getName());
    std::string usage(
        "SHOWCASE can perform AO baking on the specified glTF file. If no file is specified,"
        "it loads the most recently-used glTF file.\n"
        "Usage:\n"
        "    SHOWCASE [options] [gltf path]\n"
        "Options:\n"
        "   --help, -h\n"
        "       Prints this message\n\n"
        "   --actual-size, -s\n"
        "       Do not scale the model to fit into a unit cube\n\n"
    );
    const std::string from("SHOWCASE");
    for (size_t pos = usage.find(from); pos != std::string::npos; pos = usage.find(from, pos)) {
        usage.replace(pos, from.length(), exec_name);
    }
    std::cout << usage;
}

static int handleCommandLineArguments(int argc, char* argv[], App* app) {
    static constexpr const char* OPTSTR = "ha:i:us";
    static const struct option OPTIONS[] = {
        { "help",       no_argument,       nullptr, 'h' },
        { "actual-size", no_argument,      nullptr, 's' },
        { nullptr, 0, nullptr, 0 }
    };
    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, OPTSTR, OPTIONS, &option_index)) >= 0) {
        std::string arg(optarg ? optarg : "");
        switch (opt) {
            default:
            case 'h':
                printUsage(argv[0]);
                exit(0);
            case 's':
                app->actualSize = true;
                break;
        }
    }
    return optind;
}

static std::ifstream::pos_type getFileSize(const char* filename) {
    std::ifstream in(filename, std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

static void saveIniFile(App& app) {
    std::ofstream out(INI_FILENAME);
    out << "[recent]\n";
    out << "filename=" << app.filename.c_str() << "\n";
}

static void loadIniFile(App& app) {
    utils::Path iniPath(INI_FILENAME);
    if (!app.filename.isEmpty() || !iniPath.isFile()) {
        return;
    }
    std::ifstream infile(INI_FILENAME);
    std::string line;
    while (std::getline(infile, line)) {
        size_t sep = line.find('=');
        if (sep != std::string::npos) {
            std::string lhs = line.substr(0, sep);
            std::string rhs = line.substr(sep + 1);
            if (lhs == "filename") {
                utils::Path gltf = rhs;
                if (gltf.isFile()) {
                    app.filename = rhs;
                }
            }
        }
    }
}

static void updateOverlayVerts(App& app) {
    auto viewportSize = ImGui::GetIO().DisplaySize;
    viewportSize.x -= app.viewer->getSidebarWidth();
    OVERLAY_VERTICES[0].position.x = app.viewer->getSidebarWidth();
    OVERLAY_VERTICES[2].position.x = app.viewer->getSidebarWidth();
    OVERLAY_VERTICES[1].position.x = app.viewer->getSidebarWidth() + viewportSize.x;
    OVERLAY_VERTICES[3].position.x = app.viewer->getSidebarWidth() + viewportSize.x;
    OVERLAY_VERTICES[2].position.y = viewportSize.y;
    OVERLAY_VERTICES[3].position.y = viewportSize.y;
};

static void updateOverlay(App& app) {
    auto& rcm = app.engine->getRenderableManager();
    auto vb = app.overlayVb;
    auto ib = app.overlayIb;
    updateOverlayVerts(app);
    vb->setBufferAt(*app.engine, 0,
            VertexBuffer::BufferDescriptor(OVERLAY_VERTICES, 64, nullptr));
    rcm.destroy(app.overlayEntity);
    RenderableManager::Builder(1)
            .boundingBox({{ 0, 0, 0 }, { 1000, 1000, 1 }})
            .material(0, app.overlayMaterial)
            .geometry(0, RenderableManager::PrimitiveType::TRIANGLES, vb, ib, 0, 6)
            .culling(false)
            .receiveShadows(false)
            .castShadows(false)
            .build(*app.engine, app.overlayEntity);
}

static void updateOverlayTexture(App& app) {
    Engine& engine = *app.engine;
    using MinFilter = TextureSampler::MinFilter;
    using MagFilter = TextureSampler::MagFilter;

    // Gather information about the displayed image.
    image::LinearImage image;
    switch (app.resultsVisualization) {
        case ResultsVisualization::IMAGE_OCCLUSION: image = app.ambientOcclusion; break;
        case ResultsVisualization::IMAGE_BENT_NORMALS: image = app.bentNormals; break;
        default: return;
    }
    const int width = image.getWidth();
    const int height = image.getHeight();
    const int channels = image.getChannels();
    const void* data = image.getPixelRef();
    const Texture::InternalFormat internalFormat = channels == 1 ?
            Texture::InternalFormat::R8 : Texture::InternalFormat::RGB8;
    const Texture::Format format = channels == 1 ? Texture::Format::R : Texture::Format::RGB;

    // Create a brand new texture object if necessary.
    const Texture* tex = app.overlayTexture;
    if (!tex || tex->getWidth() != width || tex->getHeight() != height ||
            tex->getFormat() != internalFormat) {
        engine.destroy(tex);
        app.overlayTexture = Texture::Builder()
                .width(width)
                .height(height)
                .levels(1)
                .sampler(Texture::Sampler::SAMPLER_2D)
                .format(internalFormat)
                .build(engine);
        TextureSampler sampler(MinFilter::LINEAR, MagFilter::LINEAR);
        app.overlayMaterial->setParameter("luma", app.overlayTexture, sampler);
        app.overlayMaterial->setParameter("grayscale", channels == 1 ? true : false);
    }

    // Upload texture data.
    Texture::PixelBufferDescriptor buffer(data, size_t(width * height * channels * sizeof(float)),
            format, Texture::Type::FLOAT);
    app.overlayTexture->setImage(engine, 0, std::move(buffer));
}

static void createOverlay(App& app) {
    Engine& engine = *app.engine;

    static constexpr uint16_t OVERLAY_INDICES[6] = { 0, 1, 2, 3, 2, 1 };

    auto vb = VertexBuffer::Builder()
            .vertexCount(4)
            .bufferCount(1)
            .attribute(VertexAttribute::POSITION, 0, VertexBuffer::AttributeType::FLOAT2, 0, 16)
            .attribute(VertexAttribute::UV0, 0, VertexBuffer::AttributeType::FLOAT2, 8, 16)
            .build(engine);
    auto ib = IndexBuffer::Builder()
            .indexCount(6)
            .bufferType(IndexBuffer::IndexType::USHORT)
            .build(engine);
    ib->setBuffer(engine,
            IndexBuffer::BufferDescriptor(OVERLAY_INDICES, 12, nullptr));
    auto mat = Material::Builder()
            .package(RESOURCES_AOPREVIEW_DATA, RESOURCES_AOPREVIEW_SIZE)
            .build(engine);
    auto matInstance = mat->createInstance();

    app.overlayVb = vb;
    app.overlayIb = ib;
    app.overlayEntity = EntityManager::get().create();
    app.overlayMaterial = matInstance;
}

static void loadAsset(App& app, gltfio::AssetPipeline::AssetHandle handle) {

    app.currentAsset = handle;
    app.viewerAsset = app.loader->createAssetFromHandle(handle);

    // Load external textures and buffers.
    gltfio::ResourceLoader({
        .engine = app.engine,
        .gltfPath = app.filename.getAbsolutePath(),
        .normalizeSkinningWeights = true,
        .recomputeBoundingBoxes = false
    }).loadResources(app.viewerAsset);

    // Load animation data then free the source hierarchy.
    app.viewerAsset->getAnimator();
    app.state = AssetPipeline::isParameterized(handle) ? PARAMETRIZED : LOADED;

    // Destroy the old currentAsset and add the renderables to the scene.
    app.viewer->setAsset(app.viewerAsset, app.names, !app.actualSize);
}

static void loadAsset(App& app) {
    std::cout << "Loading " << app.filename << "..." << std::endl;

    if (app.filename.getExtension() == "glb") {
        std::cerr << "GLB files are not yet supported." << std::endl;
        exit(1);
    }

    // Reset the pipeline to free memory.
    auto pipeline = new gltfio::AssetPipeline();

    gltfio::AssetPipeline::AssetHandle handle = pipeline->load(app.filename);
    if (!handle) {
        delete pipeline;
        puts("Unable to load model");
        exit(1);
    }

    if (!gltfio::AssetPipeline::isFlattened(handle)) {
        handle = pipeline->flatten(handle);
        if (!handle) {
            delete pipeline;
            puts("Unable to flatten model");
            exit(1);
        }
    }

    delete app.pipeline;
    app.pipeline = pipeline;

    app.viewer->setIndirectLight(FilamentApp::get().getIBL()->getIndirectLight());

    loadAsset(app, handle);

    FilamentApp::get().setWindowTitle(app.filename.getName().c_str());
}

static void actionTestRender(App& app) {
    app.pushedState = app.state;
    app.state = RENDERING;
    app.hasTestRender = true;

    gltfio::AssetPipeline::AssetHandle currentAsset = app.currentAsset;

    // Allocate the render target for the path tracer as well as a GPU texture to display it.
    auto viewportSize = ImGui::GetIO().DisplaySize;
    viewportSize.x -= app.viewer->getSidebarWidth();
    app.ambientOcclusion = image::LinearImage(viewportSize.x, viewportSize.y, 1);
    app.resultsVisualization = ResultsVisualization::IMAGE_OCCLUSION;
    updateOverlayTexture(app);

    // Compute the camera paramaeters for the path tracer.
    // ---------------------------------------------------
    // The path tracer does not know about the top-level Filament transform that we use to fit the
    // model into a unit cube (see the -s option), so here we do little trick by temporarily
    // transforming the Filament camera before grabbing its lookAt vectors.
    auto& tcm = app.engine->getTransformManager();
    auto root = tcm.getInstance(app.viewerAsset->getRoot());
    auto cam = tcm.getInstance(app.camera->getEntity());
    filament::math::mat4f prev = tcm.getTransform(root);
    tcm.setTransform(root, inverse(prev));
    tcm.setParent(cam, root);
    filament::rays::SimpleCamera camera = {
        .aspectRatio = viewportSize.x / viewportSize.y,
        .eyePosition = app.camera->getPosition(),
        .targetPosition = app.camera->getPosition() + app.camera->getForwardVector(),
        .upVector = app.camera->getUpVector(),
        .vfovDegrees = 45, // NOTE: fov is not queryable, must match with FilamentApp
    };
    tcm.setParent(cam, {});
    tcm.setTransform(root, prev);

    // Finally, set up some callbacks and invoke the path tracer.

    using filament::math::ushort2;
    auto onRenderTile = [](ushort2, ushort2, void* userData) {
        App* app = (App*) userData;
        app->requestOverlayUpdate = true;
    };
    auto onRenderDone = [](void* userData) {
        App* app = (App*) userData;
        app->requestStatePop = true;
        app->requestOverlayUpdate = true;
    };
    app.pipeline->renderAmbientOcclusion(currentAsset, app.ambientOcclusion, camera, onRenderTile,
            onRenderDone, &app);
}

static void generateUvVisualization(const utils::Path& pngOutputPath) {
    using namespace image;
    LinearImage uvimage(256, 256, 3);
    for (int y = 0, h = uvimage.getHeight(); y < h; ++y) {
        for (int x = 0, w = uvimage.getWidth(); x < w; ++x) {
            float* dst = uvimage.getPixelRef(x, y);
            dst[0] = float(x) / w;
            dst[1] = float(y) / h;
            dst[2] = 1.0f;
        }
    }
    std::ofstream out(pngOutputPath.c_str(), std::ios::binary | std::ios::trunc);
    ImageEncoder::encode(out, ImageEncoder::Format::PNG_LINEAR, uvimage, "",
            pngOutputPath.c_str());
}

static void actionBakeAo(App& app) {
    using namespace image;
    using filament::math::ushort2;

    app.hasTestRender = false;

    auto doRender = [&app] {
        app.state = BAKING;

        // Allocate the render target for the path tracer as well as a GPU texture to display it.
        const uint32_t res = app.bakeResolution;
        app.resultsVisualization = ResultsVisualization::IMAGE_OCCLUSION;
        app.ambientOcclusion = image::LinearImage(res, res, 1);
        app.bentNormals = image::LinearImage(res, res, 3);
        app.meshNormals = image::LinearImage(res, res, 3);
        app.meshPositions = image::LinearImage(res, res, 3);
        updateOverlayTexture(app);

        auto onRenderTile = [](ushort2, ushort2, void* userData) {
            App* app = (App*) userData;
            app->requestOverlayUpdate = true;
        };

        auto onRenderDone = [](void* userData) {
            App* app = (App*) userData;
            app->requestOverlayUpdate = true;
            app->state = BAKED;

            #if DEBUG_PATHTRACER
            auto fmt = ImageEncoder::Format::PNG_LINEAR;
            std::ofstream bn("bentNormals.png", std::ios::binary | std::ios::trunc);
            image::LinearImage img = image::verticalFlip(image::vectorsToColors(app->bentNormals));
            ImageEncoder::encode(bn, fmt, img, "", "bentNormals.png");
            std::ofstream mn("meshNormals.png", std::ios::binary | std::ios::trunc);
            img = image::verticalFlip(image::vectorsToColors(app->meshNormals));
            ImageEncoder::encode(mn, fmt, img, "", "meshNormals.png");
            std::ofstream mp("meshPositions.png", std::ios::binary | std::ios::trunc);
            img = image::verticalFlip(image::vectorsToColors(app->meshPositions));
            ImageEncoder::encode(mp, fmt, img, "", "meshPositions.png");
            #endif
        };

        image::LinearImage outputs[] = {
            app.ambientOcclusion, app.bentNormals, app.meshNormals, app.meshPositions
        };
        app.pipeline->bakeAllOutputs(app.currentAsset, outputs, onRenderTile, onRenderDone, &app);

        const utils::Path folder = app.filename.getAbsolutePath().getParent();
        const utils::Path binPath = folder + "baked.bin";
        const utils::Path outPath = folder + "baked.gltf";
        const utils::Path texPath = folder + "baked.png";

        generateUvVisualization(folder + "uvs.png");

        std::ofstream out(texPath.c_str(), std::ios::binary | std::ios::trunc);
        ImageEncoder::encode(out, ImageEncoder::Format::PNG_LINEAR, app.ambientOcclusion, "",
                texPath.c_str());

        // TODO: in theory this work could be moved into the end of parameterizeJob
        app.previewAoAsset = app.pipeline->generatePreview(app.currentAsset, "baked.png");
        app.previewUvAsset = app.pipeline->generatePreview(app.currentAsset, "uvs.png");
        app.modifiedAsset = app.pipeline->replaceOcclusion(app.currentAsset, "baked.png");
    };

    auto parameterizeJob = [&app, doRender] {
        auto pipeline = new gltfio::AssetPipeline();

        app.previewAoAsset = nullptr;
        app.modifiedAsset = nullptr;
        app.previewUvAsset = nullptr;

        app.statusText = std::make_shared<std::string>("Parameterizing");
        auto parameterized = pipeline->parameterize(app.currentAsset);
        app.statusText.reset();

        if (!parameterized) {
            app.messageBoxText = std::make_shared<std::string>(
                    "Unable to parameterize mesh, check terminal output for details.");
            app.pushedState = LOADED;
            app.requestStatePop = true;
            delete pipeline;
            return;
        }

        loadAsset(app, parameterized);

        app.pushedState = PARAMETRIZED;
        app.requestStatePop = true;

        delete app.pipeline;
        app.pipeline = pipeline;

        doRender();
    };

    if (AssetPipeline::isParameterized(app.currentAsset)) {
        puts("Already parameterized.");
        doRender();
    } else {
        app.state = PARAMETRIZING;
        utils::JobSystem* js = utils::JobSystem::getJobSystem();
        utils::JobSystem::Job* parent = js->createJob();
        utils::JobSystem::Job* prep = utils::jobs::createJob(*js, parent, parameterizeJob);
        js->run(prep);
    }
}

static void actionExport(App& app) {
    const utils::Path folder = app.filename.getAbsolutePath().getParent();
    const utils::Path binPath = folder + "baked.bin";
    const utils::Path outPath = folder + "baked.gltf";
    const utils::Path texPath = folder + "baked.png";
    switch (app.exportOption) {
        case VISUALIZE_AO:
            app.pipeline->save(app.previewAoAsset, outPath, binPath);
            break;
        case VISUALIZE_UV:
            app.pipeline->save(app.previewUvAsset, outPath, binPath);
            break;
        case PRESERVE_MATERIALS:
            app.pipeline->save(app.modifiedAsset, outPath, binPath);
            break;
    }
    std::cout << "Generated " << outPath << ", " << binPath << ", and " << texPath << std::endl;
    app.state = EXPORTED;
}

int main(int argc, char** argv) {
    App app;

    app.config.title = "gltf_baker";
    app.config.iblDirectory = FilamentApp::getRootPath() + DEFAULT_IBL;
    app.requestOverlayUpdate = false;
    app.requestStatePop = false;

    int option_index = handleCommandLineArguments(argc, argv, &app);
    int num_args = argc - option_index;
    if (num_args >= 1) {
        app.filename = argv[option_index];
        if (!app.filename.exists()) {
            std::cerr << "file " << app.filename << " not found!" << std::endl;
            return 1;
        }
        if (app.filename.isDirectory()) {
            auto files = app.filename.listContents();
            for (auto file : files) {
                if (file.getExtension() == "gltf") {
                    app.filename = file;
                    break;
                }
            }
            if (app.filename.isDirectory()) {
                std::cerr << "no glTF file found in " << app.filename << std::endl;
                return 1;
            }
        }
    }

    loadIniFile(app);

    auto setup = [&](Engine* engine, View* view, Scene* scene) {
        app.engine = engine;
        app.names = new NameComponentManager(EntityManager::get());
        app.viewer = new SimpleViewer(engine, scene, view, SimpleViewer::FLAG_COLLAPSED);
        app.materials = createMaterialGenerator(engine);
        app.loader = AssetLoader::create({engine, app.materials, app.names });
        app.camera = &view->getCamera();

        if (!app.filename.isEmpty()) {
            loadAsset(app);
            saveIniFile(app);
        }

        createOverlay(app);

        app.viewer->setUiCallback([&app, scene] () {
            const ImU32 disabled = ImColor(ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            const ImU32 hovered = ImColor(ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered]);
            const ImU32 enabled = ImColor(0.5f, 0.5f, 0.0f);
            const ImVec2 buttonSize(100, 50);
            const float buttonPositions[] = { 0, buttonSize.x + 2, buttonSize.x * 2 + 3 };
            ImVec2 pos;
            ImU32 color;

            // Begin action buttons
            ImGui::GetStyle().ItemSpacing.x = 1;
            ImGui::GetStyle().FrameRounding = 10;
            ImGui::PushStyleColor(ImGuiCol_Button, enabled);
            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::BeginGroup();

            // TEST RENDER
            ImGui::SameLine(buttonPositions[0]);
            pos = ImGui::GetCursorScreenPos();
            color = ImGui::IsMouseHoveringRect(pos, pos + buttonSize) ? hovered : enabled;
            ImGui::GetWindowDrawList()->AddRectFilled(pos, pos + buttonSize, color,
                    ImGui::GetStyle().FrameRounding, ImDrawCornerFlags_Left);
            if (ImGui::Button("Test Render", buttonSize)) {
                actionTestRender(app);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Renders the asset from the current camera using a pathtracer.");
            }

            // BAKE
            ImGui::SameLine(buttonPositions[1]);
            pos = ImGui::GetCursorScreenPos();
            color = ImGui::IsMouseHoveringRect(pos, pos + buttonSize) ? hovered : enabled;
            ImGui::GetWindowDrawList()->AddRectFilled(pos, pos + buttonSize, color);
            if (ImGui::Button("Bake AO", buttonSize)) {
                actionBakeAo(app);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Generates a new set of UVs and invokes a pathtracer.");
            }

            // EXPORT
            ImGui::SameLine(buttonPositions[2]);
            pos = ImGui::GetCursorScreenPos();
            color = ImGui::IsMouseHoveringRect(pos, pos + buttonSize) ? hovered : enabled;
            const bool canExport = app.state == BAKED;
            color = canExport ? color : disabled;
            ImGui::GetWindowDrawList()->AddRectFilled(pos, pos + buttonSize, color,
                    ImGui::GetStyle().FrameRounding, ImDrawCornerFlags_Right);
            ImGui::PushStyleColor(ImGuiCol_Button, color);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, color);
            if (ImGui::Button("Export...", buttonSize) && canExport) {
                ImGui::OpenPopup("Export options");
            }
            ImGui::PopStyleColor();
            ImGui::PopStyleColor();
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Saves the baked result to disk.");
            }

            // End action buttons
            ImGui::EndGroup();
            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::PopStyleColor();
            ImGui::GetStyle().FrameRounding = 20;
            ImGui::GetStyle().ItemSpacing.x = 8;

            // Progress indicators
            if (app.statusText) {
                static const char* suffixes[] = { "...", "......", "........." };
                static float suffixAnim = 0;
                suffixAnim += 0.05f;
                const char* suffix = suffixes[int(suffixAnim) % 3];

                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {10, 10} );
                ImGui::Text("%s%s", app.statusText->c_str(), suffix);
                ImGui::PopStyleVar();
            }

            // Results
            auto addOption = [&app](const char* msg, char num, ResultsVisualization e) {
                ImGuiIO& io = ImGui::GetIO();
                int* ptr = (int*) &app.resultsVisualization;
                if (io.InputCharacters[0] == num) { app.resultsVisualization = e; }
                ImGui::RadioButton(msg, ptr, (int) e);
                ImGui::SameLine();
                ImGui::TextColored({1, 1, 0,1 }, "%c", num);
            };
            if (app.ambientOcclusion && ImGui::CollapsingHeader("Results",
                    ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Indent();
                const ResultsVisualization previousVisualization = app.resultsVisualization;
                using RV = ResultsVisualization;
                addOption("3D model with original materials", '1', RV::MESH_ORIGINAL);
                if (app.hasTestRender) {
                    addOption("Rendered AO test image", '2', RV::IMAGE_OCCLUSION);
                } else if (!app.modifiedAsset) {
                    addOption("2D texture with occlusion", '2', RV::IMAGE_OCCLUSION);
                    addOption("2D texture with bent normals", '3', RV::IMAGE_BENT_NORMALS);
                } else {
                    addOption("3D model with modified materials", '2', RV::MESH_MODIFIED);
                    addOption("3D model with new occlusion only", '3', RV::MESH_PREVIEW_AO);
                    addOption("3D model with UV visualization", '4', RV::MESH_PREVIEW_UV);
                    addOption("2D texture with occlusion", '5', RV::IMAGE_OCCLUSION);
                    addOption("2D texture with bent normals", '6', RV::IMAGE_BENT_NORMALS);
                }
                if (app.resultsVisualization != previousVisualization) {
                    app.requestOverlayUpdate = true;
                }
                ImGui::Unindent();
            }

            // Options
            if (ImGui::CollapsingHeader("Bake Options")) {
                ImGui::InputInt("Samples per pixel", &app.samplesPerPixel);
                static const int kFirstOption = std::log2(512);
                int bakeOption = std::log2(app.bakeResolution) - kFirstOption;
                ImGui::Combo("Texture size", &bakeOption,
                        "512 x 512\0"
                        "1024 x 1024\0"
                        "2048 x 2048\0");
                app.bakeResolution = 1 << (bakeOption + kFirstOption);
            }

            // Modals
            if (app.messageBoxText) {
                ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {10, 10} );
                ImGui::OpenPopup("MessageBox");
                if (ImGui::BeginPopupModal("MessageBox", nullptr,
                        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar)) {
                    ImGui::TextUnformatted(app.messageBoxText->c_str());
                    if (ImGui::Button("OK", ImVec2(120,0))) {
                        app.messageBoxText.reset();
                        ImGui::CloseCurrentPopup();
                    }
                    ImGui::EndPopup();
                }
                ImGui::PopStyleVar();
            }

            if (ImGui::BeginPopupModal("Export options", nullptr,
                    ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar)) {
                ImGui::RadioButton("Visualize ambient occlusion", (int*) &app.exportOption, 0);
                ImGui::RadioButton("Visualize generated UVs", (int*) &app.exportOption, 1);
                ImGui::RadioButton("Preserve materials", (int*) &app.exportOption, 2);
                if (ImGui::Button("OK", ImVec2(120,0))) {
                    ImGui::CloseCurrentPopup();
                    actionExport(app);
                }
                ImGui::EndPopup();
            }
        });

        // Leave FXAA enabled but we also enable MSAA for a nice result. The wireframe looks
        // much better with MSAA enabled.
        view->setSampleCount(4);
    };

    auto cleanup = [&app](Engine* engine, View*, Scene*) {
        Fence::waitAndDestroy(engine->createFence());
        delete app.viewer;
        app.materials->destroyMaterials();
        delete app.materials;
        AssetLoader::destroy(&app.loader);
        delete app.names;
    };

    auto animate = [&app](Engine* engine, View* view, double now) {
        // The baker doesn't support animation, just use frame 0.
        if (app.state != EMPTY) {
            app.viewer->applyAnimation(0.0);
        }
        bool showOverlay = app.resultsVisualization == ResultsVisualization::IMAGE_OCCLUSION ||
                app.resultsVisualization == ResultsVisualization::IMAGE_BENT_NORMALS;
        if (!app.overlayScene && showOverlay) {
            app.overlayView = FilamentApp::get().getGuiView();
            app.overlayScene = app.overlayView->getScene();
        }
        if (app.overlayScene) {
            app.overlayScene->remove(app.overlayEntity);
            if (showOverlay) {
                updateOverlay(app);
                app.overlayScene->addEntity(app.overlayEntity);
            }
        }
        if (app.requestOverlayUpdate) {
            updateOverlayTexture(app);
            app.requestOverlayUpdate = false;
        }
        if (app.requestStatePop) {
            app.state = app.pushedState;
            app.pushedState = EMPTY;
            app.requestStatePop = false;
        }
    };

    auto gui = [&app](filament::Engine* engine, filament::View* view) {
        app.viewer->updateUserInterface();
        FilamentApp::get().setSidebarWidth(app.viewer->getSidebarWidth());
    };

    FilamentApp& filamentApp = FilamentApp::get();
    filamentApp.animate(animate);

    filamentApp.setDropHandler([&] (std::string path) {
        app.viewer->removeAsset();
        app.filename = path;
        loadAsset(app);
        saveIniFile(app);
    });

    filamentApp.run(app.config, setup, cleanup, gui);

    return 0;
}
