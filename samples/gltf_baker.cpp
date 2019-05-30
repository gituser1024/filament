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

enum class ResultsVisualization : int {
    MESH_CURRENT,
    MESH_MODIFIED,
    MESH_PREVIEW_AO,
    MESH_PREVIEW_UV,
    IMAGE_OCCLUSION,
    IMAGE_BENT_NORMALS
};

struct App {
    Engine* engine = nullptr;
    Camera* camera = nullptr;
    SimpleViewer* viewer = nullptr;
    Config config;
    NameComponentManager* names = nullptr;
    MaterialProvider* materials = nullptr;
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
    bool isWorking = false;

    image::LinearImage ambientOcclusion;
    image::LinearImage bentNormals;
    image::LinearImage meshNormals;
    image::LinearImage meshPositions;

    bool viewerActualSize = false;
    utils::Path filename;
    ResultsVisualization resultsVisualization = ResultsVisualization::MESH_CURRENT;
    uint32_t bakeResolution = 1024;
    int samplesPerPixel = 256;
    ResultsVisualization exportOption = ResultsVisualization::MESH_MODIFIED;

    // Secondary threads might write to the following fields.
    std::shared_ptr<std::string> statusText;
    std::shared_ptr<std::string> messageBoxText;
    std::atomic<bool> requestViewerUpdate;

    // Export options.
    char outputFolder[PATH_SIZE];
    char gltfPath[PATH_SIZE];
    char binPath[PATH_SIZE];
    char occlusionPath[PATH_SIZE];
    char bentNormalsPath[PATH_SIZE];
};

struct OverlayVertex {
    filament::math::float2 position;
    filament::math::float2 uv;
};

static const char* DEFAULT_IBL = "envs/venetian_crossroads";
static const char* INI_FILENAME = "gltf_baker.ini";
static constexpr int PATH_SIZE = 256;

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
        "       Do not scale the model to fit into a unit cube in the viewer\n\n"
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
                app->viewerActualSize = true;
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

static void createQuadRenderable(App& app) {
    auto& rcm = app.engine->getRenderableManager();
    Engine& engine = *app.engine;

    static constexpr uint16_t OVERLAY_INDICES[6] = { 0, 1, 2, 3, 2, 1 };
    static OverlayVertex OVERLAY_VERTICES[4] = {
            {{0, 0}, {0, 0}},
            {{ 1000, 0}, {1, 0}},
            {{0,  1000}, {0, 1}},
            {{ 1000,  1000}, {1, 1}},
    };

    if (!app.overlayEntity) {
        app.overlayVb = VertexBuffer::Builder()
                .vertexCount(4)
                .bufferCount(1)
                .attribute(VertexAttribute::POSITION, 0, VertexBuffer::AttributeType::FLOAT2, 0, 16)
                .attribute(VertexAttribute::UV0, 0, VertexBuffer::AttributeType::FLOAT2, 8, 16)
                .build(engine);
        app.overlayIb = IndexBuffer::Builder()
                .indexCount(6)
                .bufferType(IndexBuffer::IndexType::USHORT)
                .build(engine);
        app.overlayIb->setBuffer(engine,
                IndexBuffer::BufferDescriptor(OVERLAY_INDICES, 12, nullptr));
        auto mat = Material::Builder()
                .package(RESOURCES_AOPREVIEW_DATA, RESOURCES_AOPREVIEW_SIZE)
                .build(engine);
        app.overlayMaterial = mat->createInstance();
        app.overlayEntity = EntityManager::get().create();
    }

    auto vb = app.overlayVb;
    auto ib = app.overlayIb;
    auto viewportSize = ImGui::GetIO().DisplaySize;
    viewportSize.x -= app.viewer->getSidebarWidth();
    OVERLAY_VERTICES[0].position.x = app.viewer->getSidebarWidth();
    OVERLAY_VERTICES[2].position.x = app.viewer->getSidebarWidth();
    OVERLAY_VERTICES[1].position.x = app.viewer->getSidebarWidth() + viewportSize.x;
    OVERLAY_VERTICES[3].position.x = app.viewer->getSidebarWidth() + viewportSize.x;
    OVERLAY_VERTICES[2].position.y = viewportSize.y;
    OVERLAY_VERTICES[3].position.y = viewportSize.y;
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

static void updateViewerMesh(App& app) {
    gltfio::AssetPipeline::AssetHandle handle;
    switch (app.resultsVisualization) {
        case ResultsVisualization::MESH_CURRENT: handle = app.currentAsset; break;
        case ResultsVisualization::MESH_MODIFIED: handle = app.modifiedAsset; break;
        case ResultsVisualization::MESH_PREVIEW_AO: handle = app.previewAoAsset; break;
        case ResultsVisualization::MESH_PREVIEW_UV: handle = app.previewUvAsset; break;
        default: return;
    }

    if (!app.viewerAsset || app.viewerAsset->getSourceAsset() != handle) {
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

        // Destroy the old currentAsset and add the renderables to the scene.
        app.viewer->setAsset(app.viewerAsset, app.names, !app.viewerActualSize);
    }
}

static void updateViewerImage(App& app) {
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
        app.overlayMaterial->setParameter("grayscale", channels == 1);
    }

    // Upload texture data.
    Texture::PixelBufferDescriptor buffer(data, size_t(width * height * channels * sizeof(float)),
            format, Texture::Type::FLOAT);
    app.overlayTexture->setImage(engine, 0, std::move(buffer));
}

static void updateViewer(App& app) {
    switch (app.resultsVisualization) {
        case ResultsVisualization::MESH_CURRENT:
        case ResultsVisualization::MESH_MODIFIED:
        case ResultsVisualization::MESH_PREVIEW_AO:
        case ResultsVisualization::MESH_PREVIEW_UV:
            updateViewerMesh(app);
            break;
        case ResultsVisualization::IMAGE_OCCLUSION:
        case ResultsVisualization::IMAGE_BENT_NORMALS:
            updateViewerImage(app);
            break;
    }
}

static void loadAssetFromDisk(App& app) {
    std::cout << "Loading " << app.filename << "..." << std::endl;
    if (app.filename.getExtension() == "glb") {
        std::cerr << "GLB files are not yet supported." << std::endl;
        exit(1);
    }

    auto pipeline = new gltfio::AssetPipeline();
    gltfio::AssetPipeline::AssetHandle handle = pipeline->load(app.filename);
    if (!handle) {
        delete pipeline;
        puts("Unable to load model");
        exit(1);
    }

    if (!gltfio::AssetPipeline::isFlattened(handle)) {
        handle = pipeline->flatten(handle, AssetPipeline::FILTER_TRIANGLES);
        if (!handle) {
            delete pipeline;
            puts("Unable to flatten model");
            exit(1);
        }
    }

    // Destroying the previous pipeline frees memory for the previously-loaded asset.
    delete app.pipeline;
    app.pipeline = pipeline;

    app.viewer->setIndirectLight(FilamentApp::get().getIBL()->getIndirectLight());
    app.currentAsset = handle;
    app.requestViewerUpdate = true;
    FilamentApp::get().setWindowTitle(app.filename.getName().c_str());
}

static void executeTestRender(App& app) {
    app.isWorking = true;
    app.hasTestRender = true;

    gltfio::AssetPipeline::AssetHandle currentAsset = app.currentAsset;

    // Allocate the render target for the path tracer as well as a GPU texture to display it.
    auto viewportSize = ImGui::GetIO().DisplaySize;
    viewportSize.x -= app.viewer->getSidebarWidth();
    app.ambientOcclusion = image::LinearImage((uint32_t) viewportSize.x, (uint32_t) viewportSize.y, 1);
    app.resultsVisualization = ResultsVisualization::IMAGE_OCCLUSION;

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
        app->requestViewerUpdate = true;
    };
    auto onRenderDone = [](void* userData) {
        App* app = (App*) userData;
        app->requestViewerUpdate = true;
        app->isWorking = false;
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

static void executeBakeAo(App& app) {
    using namespace image;
    using filament::math::ushort2;

    app.hasTestRender = false;
    app.isWorking = true;

    auto doRender = [&app] {
        const uint32_t res = app.bakeResolution;
        app.resultsVisualization = ResultsVisualization::IMAGE_OCCLUSION;
        app.ambientOcclusion = image::LinearImage(res, res, 1);
        app.bentNormals = image::LinearImage(res, res, 3);
        app.meshNormals = image::LinearImage(res, res, 3);
        app.meshPositions = image::LinearImage(res, res, 3);

        auto onRenderTile = [](ushort2, ushort2, void* userData) {
            App* app = (App*) userData;
            app->requestViewerUpdate = true;
        };

        auto onRenderDone = [](void* userData) {
            App* app = (App*) userData;
            app->requestViewerUpdate = true;

            const utils::Path folder = app->filename.getAbsolutePath().getParent();
            const utils::Path binPath = folder + "baked.bin";
            const utils::Path outPath = folder + "baked.gltf";
            const utils::Path texPath = folder + "baked.png";

            generateUvVisualization(folder + "uvs.png");

            std::ofstream out(texPath.c_str(), std::ios::binary | std::ios::trunc);
            ImageEncoder::encode(out, ImageEncoder::Format::PNG_LINEAR, app->ambientOcclusion, "",
                    texPath.c_str());

            app->previewAoAsset = app->pipeline->generatePreview(app->currentAsset, "baked.png");
            app->previewUvAsset = app->pipeline->generatePreview(app->currentAsset, "uvs.png");
            app->modifiedAsset = app->pipeline->replaceOcclusion(app->currentAsset, "baked.png");
            app->isWorking = false;
        };

        image::LinearImage outputs[] = {
            app.ambientOcclusion, app.bentNormals, app.meshNormals, app.meshPositions
        };
        app.pipeline->bakeAllOutputs(app.currentAsset, outputs, onRenderTile, onRenderDone, &app);
    };

    auto parameterizeJob = [&app, doRender] {
        app.previewAoAsset = nullptr;
        app.modifiedAsset = nullptr;
        app.previewUvAsset = nullptr;
        app.ambientOcclusion = LinearImage();
        app.statusText = std::make_shared<std::string>("Parameterizing");
        auto parameterized = app.pipeline->parameterize(app.currentAsset);
        app.statusText.reset();
        if (!parameterized) {
            app.messageBoxText = std::make_shared<std::string>(
                    "Unable to parameterize mesh, check terminal output for details.");
            app.isWorking = false;
            return;
        }
        app.currentAsset = parameterized;
        app.requestViewerUpdate = true;
        doRender();
    };

    if (AssetPipeline::isParameterized(app.currentAsset)) {
        puts("Already parameterized.");
        doRender();
    } else {
        utils::JobSystem* js = utils::JobSystem::getJobSystem();
        utils::JobSystem::Job* parent = js->createJob();
        utils::JobSystem::Job* prep = utils::jobs::createJob(*js, parent, parameterizeJob);
        js->run(prep);
    }
}

static void executeExport(App& app) {
    const utils::Path folder = app.filename.getAbsolutePath().getParent();
    const utils::Path binPath = folder + "baked.bin";
    const utils::Path outPath = folder + "baked.gltf";
    const utils::Path texPath = folder + "baked.png";
    using RV = ResultsVisualization;
    switch (app.exportOption) {
        case ResultsVisualization::MESH_CURRENT:
            app.pipeline->save(app.currentAsset, outPath, binPath);
            break;
        case ResultsVisualization::MESH_MODIFIED:
            app.pipeline->save(app.modifiedAsset, outPath, binPath);
            break;
        case ResultsVisualization::MESH_PREVIEW_AO:
            app.pipeline->save(app.previewAoAsset, outPath, binPath);
            break;
        case ResultsVisualization::IMAGE_OCCLUSION:
            // TODO
            break;
        case ResultsVisualization::IMAGE_BENT_NORMALS:
            // TODO
            break;
        default:
            return;
    }
    std::cout << "Generated " << outPath << ", " << binPath << ", and " << texPath << std::endl;
}

int main(int argc, char** argv) {
    App app;

    strncpy(app.gltfPath, "baked.gltf", PATH_SIZE);
    strncpy(app.binPath, "baked.bin", PATH_SIZE);
    strncpy(app.occlusionPath, "occlusion.png", PATH_SIZE);
    strncpy(app.bentNormalsPath, "bentNormals.png", PATH_SIZE);

    app.config.title = "gltf_baker";
    app.config.iblDirectory = FilamentApp::getRootPath() + DEFAULT_IBL;
    app.requestViewerUpdate = false;

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
            for (const auto& file : files) {
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

    const utils::Path defaultFolder = app.filename.getAbsolutePath().getParent();
    strncpy(app.outputFolder, defaultFolder.c_str(), sizeof(app.outputFolder));

    loadIniFile(app);

    auto setup = [&](Engine* engine, View* view, Scene* scene) {
        app.engine = engine;
        app.names = new NameComponentManager(EntityManager::get());

        const int kInitialSidebarWidth = 322;
        app.viewer = new SimpleViewer(engine, scene, view, SimpleViewer::FLAG_COLLAPSED,
                kInitialSidebarWidth);
        app.viewer->enableSunlight(false);
        app.viewer->enableSSAO(false);
        app.viewer->setIBLIntensity(50000.0f);

        app.materials = createMaterialGenerator(engine);
        app.loader = AssetLoader::create({engine, app.materials, app.names });
        app.camera = &view->getCamera();

        if (!app.filename.isEmpty()) {
            loadAssetFromDisk(app);
            saveIniFile(app);
        }

        app.viewer->setUiCallback([&app, scene] () {
            const ImU32 disabledColor = ImColor(ImGui::GetStyle().Colors[ImGuiCol_TextDisabled]);
            const ImU32 hoveredColor = ImColor(ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered]);
            const ImU32 enabledColor = ImColor(0.5f, 0.5f, 0.0f);
            const ImVec2 buttonSize(100, 50);
            const float buttonPositions[] = { 0, 2 + buttonSize.x, 4 + buttonSize.x * 2 };
            ImVec2 pos;
            ImU32 color;
            bool enabled;

            // Begin action buttons
            ImGui::GetStyle().ItemSpacing.x = 1;
            ImGui::GetStyle().FrameRounding = 10;
            ImGui::PushStyleColor(ImGuiCol_Button, enabledColor);
            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::BeginGroup();

            using OnClick = void(*)(App& app);
            auto showActonButton = [&](const char* label, int cornerFlags, OnClick fn) {
                pos = ImGui::GetCursorScreenPos();
                color = enabled ? enabledColor : disabledColor;
                color = ImGui::IsMouseHoveringRect(pos, pos + buttonSize) ? hoveredColor : color;
                ImGui::GetWindowDrawList()->AddRectFilled(pos, pos + buttonSize, color,
                        ImGui::GetStyle().FrameRounding, cornerFlags);
                ImGui::PushStyleColor(ImGuiCol_Button, color);
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered, color);
                if (ImGui::Button(label, buttonSize) && enabled) {
                    fn(app);
                }
                ImGui::PopStyleColor();
                ImGui::PopStyleColor();
            };

            // TEST RENDER
            ImGui::SameLine(buttonPositions[0]);
            enabled = !app.isWorking;
            showActonButton("Test Render", ImDrawCornerFlags_Left, [](App& app) {
                executeTestRender(app);
            });
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Renders the asset from the current camera using a pathtracer.");
            }

            // BAKE
            ImGui::SameLine(buttonPositions[1]);
            enabled = !app.isWorking;
            showActonButton("Bake AO", 0, [](App& app) {
                executeBakeAo(app);
            });
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Generates a new set of UVs and invokes a pathtracer.");
            }

            // EXPORT
            ImGui::SameLine(buttonPositions[2]);
            enabled = !app.isWorking && !app.hasTestRender && app.modifiedAsset;
            showActonButton("Export...", ImDrawCornerFlags_Right, [](App& app) {
                ImGui::OpenPopup("Export options");
            });
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
                addOption("3D model with original materials", '1', RV::MESH_CURRENT);
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
                    app.requestViewerUpdate = true;
                }
                ImGui::Unindent();
            }

            // Options
            if (ImGui::CollapsingHeader("Bake Options")) {
                ImGui::InputInt("Samples per pixel", &app.samplesPerPixel);
                static const int kFirstOption = (int) std::log2(512);
                int bakeOption = (int) std::log2(app.bakeResolution) - kFirstOption;
                ImGui::Combo("Texture size", &bakeOption,
                        "512 x 512\0"
                        "1024 x 1024\0"
                        "2048 x 2048\0");
                app.bakeResolution = 1u << uint32_t(bakeOption + kFirstOption);
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
                    ImGuiWindowFlags_AlwaysAutoResize)) {
                using RV = ResultsVisualization;

                ImGui::InputText("Folder", app.outputFolder, PATH_SIZE);
                ImGui::InputText("glTF", app.gltfPath, PATH_SIZE);
                ImGui::InputText("glTF sidecar bin", app.binPath, PATH_SIZE);
                ImGui::InputText("Occlusion image", app.occlusionPath, PATH_SIZE);
                ImGui::InputText("Bent normals image", app.bentNormalsPath, PATH_SIZE);

                auto radio = [&app](const char* name, RV value) {
                    int* ptr = (int*) &app.exportOption;
                    ImGui::RadioButton(name, ptr, (int) value);
                };
                radio("Export the flattened glTF with original materials", RV::MESH_CURRENT);
                radio("Export the flattened glTF with modified materials", RV::MESH_MODIFIED);
                radio("Export the flattened glTF with new occlusion only", RV::MESH_PREVIEW_AO);
                radio("Export only the occlusion and bent normals images", RV::IMAGE_OCCLUSION);
                if (ImGui::Button("OK", ImVec2(120,0))) {
                    ImGui::CloseCurrentPopup();
                    executeExport(app);
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
        if (app.viewerAsset) {
            app.viewer->applyAnimation(0.0);
        }

        app.overlayView = FilamentApp::get().getGuiView();
        app.overlayScene = app.overlayView->getScene();
        app.overlayScene->remove(app.overlayEntity);

        // Update the overlay quad geometry just in case the window size changed.
        const bool showOverlay = app.resultsVisualization == ResultsVisualization::IMAGE_OCCLUSION
                || app.resultsVisualization == ResultsVisualization::IMAGE_BENT_NORMALS;
        if (showOverlay) {
            createQuadRenderable(app);
            app.overlayScene->addEntity(app.overlayEntity);
        }

        // If requested update the overlay quad texture or 3D mesh data.
        if (app.requestViewerUpdate) {
            updateViewer(app);
            app.requestViewerUpdate = false;
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
        const utils::Path defaultFolder = app.filename.getAbsolutePath().getParent();
        strncpy(app.outputFolder, defaultFolder.c_str(), PATH_SIZE);
        loadAssetFromDisk(app);
        saveIniFile(app);
    });

    filamentApp.run(app.config, setup, cleanup, gui);

    return 0;
}
