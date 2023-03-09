#include "GLFW/glfw3.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"
#include "surface_functions.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include <iostream>
#include "MathUtilities.cuh"
#include "WorldData.cuh"
#include "random"

#define REFERENCE_PATH_TRACING 0
#define INGAME_PATH_TRACING 1
#define UNLIT 2

#define RENDER_MODE 1

Player player = { { 30.5, 28.5, 20.5 }, { 1.0, ZERO_VEC3D }, TAU * 0.25 }; // 32.5, 19.0, 15.0 and 32.0, 32.0, 10.0

GLuint screenGLTexture;
cudaGraphicsResource_t screenCudaResource;

int globalBlockUpdate = 0;

int cacheSizes[BLOCK_TYPES];

int cachedFaces = 0;

//surface<void, cudaSurfaceType2D> 
int accumulatedFrameCount = 0;
int frameCount = 0;

std::random_device seedEngine;

/*__device__ void GetColorAtFace(Vec3D* color, GBuffer gBuffer, WorldData worldData, Vec3D currentPos, Vec3D centerPos, Vec3D playerPos, int normalId, Vec3D normal, Matrix3X3 rotationMatrix, float tanHalfFOV)
{
    int intRayPosX = int(currentPos.x);
    int intRayPosY = int(currentPos.y);
    int intRayPosZ = int(currentPos.z);

    int chunkIndex = (intRayPosZ / CHUNK_SIZE) * (WORLD_H / CHUNK_SIZE) * (WORLD_W / CHUNK_SIZE) + (intRayPosY / CHUNK_SIZE) * (WORLD_W / CHUNK_SIZE) + (intRayPosX / CHUNK_SIZE);

    Block block = worldData.blocks[chunkIndex * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + (intRayPosZ % CHUNK_SIZE) * CHUNK_SIZE * CHUNK_SIZE + (intRayPosY % CHUNK_SIZE) * CHUNK_SIZE + (intRayPosX % CHUNK_SIZE)];
    int blockId = block.id;

    if (blockId != 0)
    {
        int voxelX = (currentPos.x - intRayPosX) * BLOCK_RES;
        int voxelY = (currentPos.y - intRayPosY) * BLOCK_RES;
        int voxelZ = (currentPos.z - intRayPosZ) * BLOCK_RES;

        int voxelArrayPos = blockId * BLOCK_RES * BLOCK_RES * BLOCK_RES + voxelZ * BLOCK_RES * BLOCK_RES + voxelY * BLOCK_RES + voxelX;

        int irradianceCachePos = block.irradianceCacheOffset + worldData.cachePosLUT[voxelArrayPos * VOXEL_FACES + normalId - 1];

        if (irradianceCachePos != -1)
        {
            *color = worldData.irradianceCache[irradianceCachePos].irradiance * FastReciprocal(worldData.irradianceCache[irradianceCachePos].bounceCount);
        }
    }
}*/

__global__ void DrawFrame(GBuffer gBuffer, WorldData worldData, Vec3D playerPos, Matrix3X3 rotationMatrix, float tanHalfFOV, cudaSurfaceObject_t screenCudaSurfaceObject, const int screenCellW, const int screenCellH)
{
    const int screenCellX = (blockIdx.x * blockDim.x + threadIdx.x) * screenCellW;
    const int screenCellY = (blockIdx.y * blockDim.y + threadIdx.y) * screenCellH;

    for (int y = screenCellY; y < screenCellY + screenCellH; ++y)
    {
        for (int x = screenCellX; x < screenCellX + screenCellW; ++x)
        {
            int pixelNumber = y * SCREEN_W + x;

            Vec3D color = ZERO_VEC3D;

            /*Vec3D albedo = {1.0, 1.0, 1.0}; // gBuffer.albedoBuffer[pixelNumber];

            int normalId = gBuffer.normalBuffer[pixelNumber];

            Vec3D position = gBuffer.positionBuffer[pixelNumber];

            float voxelPosX = int(position.x * BLOCK_RES) / float(BLOCK_RES);
            float voxelPosY = int(position.y * BLOCK_RES) / float(BLOCK_RES);
            float voxelPosZ = int(position.z * BLOCK_RES) / float(BLOCK_RES);

            Vec3D voxelPos = { voxelPosX, voxelPosY, voxelPosZ };

            Vec3D relativePos = (position - voxelPos) * BLOCK_RES;

            Matrix3X3 tangentMatrix = gBuffer.tangentMatrixBuffer[pixelNumber];

            float u = Dot(relativePos, tangentMatrix.iHat);
            float v = Dot(relativePos, tangentMatrix.kHat);

            Vec3D interpolationAxis1 = tangentMatrix.iHat;

            if (u < 0.5)
            {
                Scale(&interpolationAxis1, -1);
            }

            Vec3D interpolationAxis2 = tangentMatrix.kHat;

            if (v < 0.5)
            {
                Scale(&interpolationAxis2, -1);
            }

            //Vec3D color = gBuffer.frameBuffer[pixelNumber];*/

            int primaryIrradianceCachePos = gBuffer.primaryIrradianceCachePositions[pixelNumber];

            if (primaryIrradianceCachePos != -1)
            {
                Vec3D irradiance = worldData.irradianceCache[primaryIrradianceCachePos].irradiance;

                int bounceCount = worldData.irradianceCache[primaryIrradianceCachePos].bounceCount;

                color = irradiance * FastReciprocal(bounceCount);

                /*Vec3D faceColor1 = irradiance * FastReciprocal(bounceCount);

                Vec3D facePos1 = voxelPos + (0.5f / BLOCK_RES) * tangentMatrix.iHat + (0.5f / BLOCK_RES) * tangentMatrix.kHat - tangentMatrix.jHat * 0.02;


                Vec3D faceColor2 = faceColor1;

                Vec3D facePos2 = facePos1 + (1.0f / BLOCK_RES) * interpolationAxis1;

                GetColorAtFace(&faceColor2, gBuffer, worldData, facePos2, facePos1, playerPos, normalId, tangentMatrix.jHat, rotationMatrix, tanHalfFOV);


                Vec3D faceColor3 = faceColor1;

                Vec3D facePos3 = facePos1 + (1.0f / BLOCK_RES) * interpolationAxis2;

                GetColorAtFace(&faceColor3, gBuffer, worldData, facePos3, facePos1, playerPos, normalId, tangentMatrix.jHat, rotationMatrix, tanHalfFOV);


                Vec3D faceColor4 = faceColor1;

                Vec3D facePos4 = facePos1 + (1.0f / BLOCK_RES) * (interpolationAxis1 + interpolationAxis2);

                GetColorAtFace(&faceColor4, gBuffer, worldData, facePos4, facePos1, playerPos, normalId, tangentMatrix.jHat, rotationMatrix, tanHalfFOV);

                //faceColor2 = ZERO_VEC3D;
                //faceColor3 = ZERO_VEC3D;
                //faceColor4 = ZERO_VEC3D;

                Vec3D posRelativeToFirstFace = (position - facePos1) * BLOCK_RES;

                float t1 = Dot(posRelativeToFirstFace, interpolationAxis1);
                float t2 = Dot(posRelativeToFirstFace, interpolationAxis2);

                Vec3D interpolatedColor1 = Lerp(faceColor1, faceColor2, t1);
                Vec3D interpolatedColor2 = Lerp(faceColor3, faceColor4, t1);

                color = Lerp(interpolatedColor1, interpolatedColor2, t2);*/


                float maxLightComponent = Max(Max(color.x, Max(color.y, color.z)), 0.01);

                float newMaxLightComponent = bounceCount * 0.1;

                float lightScalar = FastDivide(Min(maxLightComponent, newMaxLightComponent), maxLightComponent);

                color = color * lightScalar;

                Vec3D specularColor = gBuffer.specularBuffer[pixelNumber];
                float specularity = gBuffer.specularityBuffer[pixelNumber];

                color = color * (1 - specularity) + specularity * specularColor;
            }

            color = ToneMapper(color);

            color = { Min(color.x, 1), Min(color.y, 1), Min(color.z, 1) };

            color.x = GammaCorrection(color.x);
            color.y = GammaCorrection(color.y);
            color.z = GammaCorrection(color.z);

            color = color * 255;

            surf2Dwrite(make_uchar4(color.x, color.y, color.z, 255), screenCudaSurfaceObject, x * sizeof(uchar4), y);
        }
    }
}

__global__ void DeamplifyVoxels(GBuffer gBuffer, WorldData worldData, const int screenCellW, const int screenCellH, int globalBlockUpdate)
{
    const int screenCellX = (blockIdx.x * blockDim.x + threadIdx.x) * screenCellW;
    const int screenCellY = (blockIdx.y * blockDim.y + threadIdx.y) * screenCellH;

    for (int y = screenCellY; y < screenCellY + screenCellH; ++y)
    {
        for (int x = screenCellX; x < screenCellX + screenCellW; ++x)
        {
            int pixelNumber = y * SCREEN_W + x;

            int primaryIrradianceCachePos = gBuffer.primaryIrradianceCachePositions[pixelNumber];

            if (primaryIrradianceCachePos != -1)
            {
                int currentUpdate = worldData.irradianceCache[primaryIrradianceCachePos].currentUpdate;

                if (currentUpdate != globalBlockUpdate)
                {
                    /*Vec3D irradiance = gBuffer.irradianceBuffer[pixelNumber].irradiance;
                    int bounceCount = gBuffer.irradianceBuffer[pixelNumber].bounceCount;

                    float scalar = FastDivide(Min(DEAMPLIFIED_BOUNCE_COUNT, bounceCount), bounceCount);

                    float maxIrradianceComponent = Max(irradiance.x, Max(irradiance.y, irradiance.z));

#define IRRADIANCE_CLAMP 1

                    if (maxIrradianceComponent * scalar > IRRADIANCE_CLAMP)
                    {
                        scalar = FastDivide(IRRADIANCE_CLAMP, maxIrradianceComponent);
                    }

                    worldData.irradianceCache[primaryIrradianceCachePos].irradiance = irradiance * scalar;
                    worldData.irradianceCache[primaryIrradianceCachePos].bounceCount = Max(bounceCount * scalar, 1);
                    worldData.irradianceCache[primaryIrradianceCachePos].currentUpdate = globalBlockUpdate;*/

                    worldData.irradianceCache[primaryIrradianceCachePos].irradiance = ZERO_VEC3D;
                    worldData.irradianceCache[primaryIrradianceCachePos].bounceCount = 1;
                    worldData.irradianceCache[primaryIrradianceCachePos].currentUpdate = globalBlockUpdate;
                }
            }
        }
    }
}

__global__ void AccumulateVoxels(GBuffer gBuffer, WorldData worldData, const int screenCellW, const int screenCellH)
{
    const int screenCellX = (blockIdx.x * blockDim.x + threadIdx.x) * screenCellW;
    const int screenCellY = (blockIdx.y * blockDim.y + threadIdx.y) * screenCellH;

    for (int y = screenCellY; y < screenCellY + screenCellH; ++y)
    {
        for (int x = screenCellX; x < screenCellX + screenCellW; ++x)
        {
            int pixelNumber = y * SCREEN_W + x;

            int primaryIrradianceCachePos = gBuffer.primaryIrradianceCachePositions[pixelNumber];

            if (primaryIrradianceCachePos != -1)
            {
                int bounceCount = worldData.irradianceCache[primaryIrradianceCachePos].bounceCount;
                bool updateVoxel = gBuffer.updateVoxel[pixelNumber];

                if (bounceCount <= MAX_IRRADIANCE_SAMPLES && updateVoxel)
                {
                    Vec3D color = gBuffer.frameBuffer[pixelNumber];

                    atomicAdd(&(worldData.irradianceCache[primaryIrradianceCachePos].irradiance.x), color.x);
                    atomicAdd(&(worldData.irradianceCache[primaryIrradianceCachePos].irradiance.y), color.y);
                    atomicAdd(&(worldData.irradianceCache[primaryIrradianceCachePos].irradiance.z), color.z);
                    atomicAdd(&(worldData.irradianceCache[primaryIrradianceCachePos].bounceCount), 1);
                }
            }
        }
    }
}

__global__ void AtrousFilter(Vec3D* filteredBuffer, Vec3D* frameBuffer, uint8_t* normalBuffer, Vec3D* positionBuffer, int stepSize, const int imageW, const int imageH, const int screenCellW, const int screenCellH, const bool rotateImage = false)
{
    int screenCellX = (blockIdx.x * blockDim.x + threadIdx.x) * screenCellW;
    int screenCellY = (blockIdx.y * blockDim.y + threadIdx.y) * screenCellH;

#define KERNEL_WIDTH 5

    float kernel[KERNEL_WIDTH] =
    {
        7, 26, 41, 26,  7
    };

    Vec3D normalTable[7] =
    {
        { -1, 0, 0 },
        { 1, 0, 0 },
        { 0, -1, 0 },
        { 0, 1, 0 },
        { 0, 0, -1 },
        { 0, 0, 1 },
        { 0, 0, 0 }
    };

    if (rotateImage)
    {
        for (int y = screenCellY; y < screenCellY + screenCellH; ++y)
        {
            for (int x = screenCellX; x < screenCellX + screenCellW; ++x)
            {
                int centerPixelNumber = y * imageW + x;

                int centerPixelNormalId = normalBuffer[centerPixelNumber];
                Vec3D centerPixelNormal = normalTable[centerPixelNormalId];
                Vec3D centerPixelWorldPos = positionBuffer[centerPixelNumber];

                Vec3D filteredPixel = ZERO_VEC3D;
                float normalizationWeight = 0;

                for (int i = 0; i < KERNEL_WIDTH; ++i)
                {
                    int pixelX = x + (i - KERNEL_WIDTH / 2) * stepSize;

                    int kernelPixelNumber = y * imageW + pixelX;

                    int kernelPixelNormalId = normalBuffer[kernelPixelNumber];
                    Vec3D kernelPixelNormal = normalTable[kernelPixelNormalId];
                    Vec3D kernelPixelWorldPos = positionBuffer[kernelPixelNumber];

                    float normalsWeight = centerPixelNormalId == kernelPixelNormalId;

                    Vec3D positionDifference = kernelPixelWorldPos - centerPixelWorldPos;

#define DEPTH_CUTOFF 0.05

                    float positionWeight = Abs(Dot(positionDifference, centerPixelNormal)) < DEPTH_CUTOFF;

                    if (pixelX >= 0 && pixelX < imageW && normalsWeight == 1 && positionWeight == 1)
                    {
                        float weight = kernel[i];

                        &filteredPixel += frameBuffer[kernelPixelNumber] * weight;

                        normalizationWeight += weight;
                    }
                }

                Scale(&filteredPixel, FastReciprocal(normalizationWeight));

                filteredBuffer[x * imageH + y] = filteredPixel;
            }
        }
    }
    else
    {
        for (int y = screenCellY; y < screenCellY + screenCellH; ++y)
        {
            for (int x = screenCellX; x < screenCellX + screenCellW; ++x)
            {
                int centerPixelNumber = y * imageW + x;

                int centerPixelNormalId = normalBuffer[centerPixelNumber];
                Vec3D centerPixelNormal = normalTable[centerPixelNormalId];
                Vec3D centerPixelWorldPos = positionBuffer[centerPixelNumber];

                Vec3D filteredPixel = ZERO_VEC3D;
                float normalizationWeight = 0;

                for (int i = 0; i < KERNEL_WIDTH; ++i)
                {
                    int pixelX = x + (i - KERNEL_WIDTH / 2) * stepSize;

                    int kernelPixelNumber = y * imageW + pixelX;

                    int kernelPixelNormalId = normalBuffer[kernelPixelNumber];
                    Vec3D kernelPixelNormal = normalTable[kernelPixelNormalId];
                    Vec3D kernelPixelWorldPos = positionBuffer[kernelPixelNumber];

                    float normalsWeight = centerPixelNormalId == kernelPixelNormalId;

                    Vec3D positionDifference = kernelPixelWorldPos - centerPixelWorldPos;

#define DEPTH_CUTOFF 0.05

                    float positionWeight = Abs(Dot(positionDifference, centerPixelNormal)) < DEPTH_CUTOFF;

                    if (pixelX >= 0 && pixelX < imageW && normalsWeight == 1 && positionWeight == 1)
                    {
                        float weight = kernel[i];

                        &filteredPixel += frameBuffer[kernelPixelNumber] * weight;

                        normalizationWeight += weight;
                    }
                }

                Scale(&filteredPixel, FastReciprocal(normalizationWeight));

                filteredBuffer[centerPixelNumber] = filteredPixel;
            }
        }
    }
}

__device__ Vec3D SampleLight(Vec3D rayPos, Vec3D rayDir, float maxDistance, Block* blocks, Voxel* blockTypes)
{
    float reciprocalRayX = FastReciprocal(rayDir.x);
    float reciprocalRayY = FastReciprocal(rayDir.y);
    float reciprocalRayZ = FastReciprocal(rayDir.z);

    #define OFFSET 0.0001

    float heavysideX = (rayDir.x >= 0) * (1 + OFFSET) - OFFSET; // accounting for dumb floating point errors
    float heavysideY = (rayDir.y >= 0) * (1 + OFFSET) - OFFSET;
    float heavysideZ = (rayDir.z >= 0) * (1 + OFFSET) - OFFSET;

    float signX = (rayDir.x >= 0) * 2 - 1;
    float signY = (rayDir.y >= 0) * 2 - 1;
    float signZ = (rayDir.z >= 0) * 2 - 1;

    float manhattanRayLength = Max(Abs(rayDir.x), Max(Abs(rayDir.y), Abs(rayDir.z)));

    float scaleX = 0;
    float scaleY = 0;
    float scaleZ = 0;

    float scale = 0;

    float distance = 0; // manhattan distance, fyi

    while (distance < maxDistance)
    {
        if (rayPos.x < 0 || rayPos.x >= WORLD_W || rayPos.y < 0 || rayPos.y >= WORLD_H || rayPos.z < 0 || rayPos.z >= WORLD_L)
        {
            return ZERO_VEC3D;
        }

        int intRayPosX = int(rayPos.x);
        int intRayPosY = int(rayPos.y);
        int intRayPosZ = int(rayPos.z);

        int chunkIndex = (intRayPosZ / CHUNK_SIZE) * (WORLD_H / CHUNK_SIZE) * (WORLD_W / CHUNK_SIZE) + (intRayPosY / CHUNK_SIZE) * (WORLD_W / CHUNK_SIZE) + (intRayPosX / CHUNK_SIZE);

        Block block = blocks[chunkIndex * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + (intRayPosZ % CHUNK_SIZE) * CHUNK_SIZE * CHUNK_SIZE + (intRayPosY % CHUNK_SIZE) * CHUNK_SIZE + (intRayPosX % CHUNK_SIZE)];
        int blockId = block.id;

        int voxelX = (rayPos.x - intRayPosX) * BLOCK_RES;
        int voxelY = (rayPos.y - intRayPosY) * BLOCK_RES;
        int voxelZ = (rayPos.z - intRayPosZ) * BLOCK_RES;

        int voxelArrayPos = int(blockId * BLOCK_RES * BLOCK_RES * BLOCK_RES + voxelZ * BLOCK_RES * BLOCK_RES + voxelY * BLOCK_RES + voxelX);

        Voxel voxel = blockTypes[voxelArrayPos];
        uchar4 voxelAlbedo = voxel.albedo;

        if (voxelAlbedo.w > 0)
        {
            uchar4 voxelAlbedo = voxel.albedo;
            Vec3D diffuseTint = { voxelAlbedo.x / 255.0f, voxelAlbedo.y / 255.0f, voxelAlbedo.z / 255.0f };

            return diffuseTint * voxel.emittance;
        }

        float addedDist = block.distance;

        float cellSize = 1.0f / BLOCK_RES;
        float reciprocalCellSize = BLOCK_RES;

        if (blockId == 0)
        {
            cellSize = CHUNK_SIZE;
            reciprocalCellSize = 1.0f / CHUNK_SIZE;
        }

        scaleX = ((int(rayPos.x * reciprocalCellSize) + heavysideX) * cellSize - rayPos.x) * reciprocalRayX;
        scaleY = ((int(rayPos.y * reciprocalCellSize) + heavysideY) * cellSize - rayPos.y) * reciprocalRayY;
        scaleZ = ((int(rayPos.z * reciprocalCellSize) + heavysideZ) * cellSize - rayPos.z) * reciprocalRayZ;

        if (blockId == 0)
        {
            // how much rayDir should be scaled to intersect the nearest xyz grid planes respectively

            scaleX = Min((intRayPosX + heavysideX - rayPos.x + addedDist * signX) * reciprocalRayX, scaleX);
            scaleY = Min((intRayPosY + heavysideY - rayPos.y + addedDist * signY) * reciprocalRayY, scaleY);
            scaleZ = Min((intRayPosZ + heavysideZ - rayPos.z + addedDist * signZ) * reciprocalRayZ, scaleZ);
        }

        scale = Min(scaleX, Min(scaleY, scaleZ));

        Vec3D step = rayDir * scale;

        &rayPos += step;

        distance += manhattanRayLength * scale;
    }

    return ZERO_VEC3D;
}

__device__ bool Raycast(Vec3D rayPos, Vec3D rayDir, float maxDistance, Block* blocks, Voxel* blockTypes, Vec3D* newRayPos, Block* hitBlock, Voxel* hitVoxel, int* hitVoxelArrayPos, Matrix3X3* newTangentMatrix, float* distanceTravelled)
{
    float reciprocalRayX = FastReciprocal(rayDir.x);
    float reciprocalRayY = FastReciprocal(rayDir.y);
    float reciprocalRayZ = FastReciprocal(rayDir.z);

    #define OFFSET 0.0001 // 0.0001

    float heavysideX = (rayDir.x >= 0) * (1 + OFFSET) - OFFSET; // accounting for dumb floating point errors
    float heavysideY = (rayDir.y >= 0) * (1 + OFFSET) - OFFSET;
    float heavysideZ = (rayDir.z >= 0) * (1 + OFFSET) - OFFSET;

    float signX = (rayDir.x >= 0) * 2 - 1;
    float signY = (rayDir.y >= 0) * 2 - 1;
    float signZ = (rayDir.z >= 0) * 2 - 1;

    float manhattanRayLength = Max(Abs(rayDir.x), Max(Abs(rayDir.y), Abs(rayDir.z)));

    float distance = 0; // manhattan distance, fyi

    Matrix3X3 tangentMatrix; // specifices the orientation of the surface that was hit. The normal and tangents

    float scaleX = 0;
    float scaleY = 0;
    float scaleZ = 0;

    float scale = 0;

    while (distance < maxDistance)
    {
        if (rayPos.x < 0 || rayPos.x >= WORLD_W || rayPos.y < 0 || rayPos.y >= WORLD_H || rayPos.z < 0 || rayPos.z >= WORLD_L)
        {
            return false;
        }

        int intRayPosX = int(rayPos.x);
        int intRayPosY = int(rayPos.y);
        int intRayPosZ = int(rayPos.z);

        int chunkIndex = (intRayPosZ / CHUNK_SIZE) * (WORLD_H / CHUNK_SIZE) * (WORLD_W / CHUNK_SIZE) + (intRayPosY / CHUNK_SIZE) * (WORLD_W / CHUNK_SIZE) + (intRayPosX / CHUNK_SIZE);

        Block block = blocks[chunkIndex * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + (intRayPosZ % CHUNK_SIZE) * CHUNK_SIZE * CHUNK_SIZE + (intRayPosY % CHUNK_SIZE) * CHUNK_SIZE + (intRayPosX % CHUNK_SIZE)];
        int blockId = block.id;

        int voxelX = (rayPos.x - intRayPosX) * BLOCK_RES;
        int voxelY = (rayPos.y - intRayPosY) * BLOCK_RES;
        int voxelZ = (rayPos.z - intRayPosZ) * BLOCK_RES;

        int voxelArrayPos = blockId * BLOCK_RES * BLOCK_RES * BLOCK_RES + voxelZ * BLOCK_RES * BLOCK_RES + voxelY * BLOCK_RES + voxelX;

        Voxel voxel = blockTypes[voxelArrayPos];
        uchar4 voxelAlbedo = voxel.albedo;

        if (voxelAlbedo.w > 0)
        {
            // return hit information

            if (scale == scaleX)
            {
                tangentMatrix.iHat = { 0, 1, 0 };
                tangentMatrix.jHat = { -signX, 0, 0 };
                tangentMatrix.kHat = { 0, 0, 1 };
            }
            else if (scale == scaleY)
            {
                tangentMatrix.iHat = { 1, 0, 0 };
                tangentMatrix.jHat = { 0, -signY, 0 };
                tangentMatrix.kHat = { 0, 0, 1 };
            }
            else if (scale == scaleZ)
            {
                tangentMatrix.iHat = { 1, 0, 0 };
                tangentMatrix.jHat = { 0, 0, -signZ };
                tangentMatrix.kHat = { 0, 1, 0 };
            }

            &rayPos += tangentMatrix.jHat * 2 * 0.01; // offset the position to avoid self intersections in later raycasts

            *newRayPos = rayPos;
            *hitBlock = block;
            *hitVoxel = voxel;
            *hitVoxelArrayPos = voxelArrayPos;
            *newTangentMatrix = tangentMatrix;
            *distanceTravelled += distance;

            return true;
        }

        float addedDist = block.distance;

        float cellSize = 1.0f / BLOCK_RES;
        float reciprocalCellSize = BLOCK_RES;

        if (blockId == 0)
        {
            cellSize = CHUNK_SIZE;
            reciprocalCellSize = 1.0f / CHUNK_SIZE;
        }

        scaleX = ((int(rayPos.x * reciprocalCellSize) + heavysideX) * cellSize - rayPos.x) * reciprocalRayX;
        scaleY = ((int(rayPos.y * reciprocalCellSize) + heavysideY) * cellSize - rayPos.y) * reciprocalRayY;
        scaleZ = ((int(rayPos.z * reciprocalCellSize) + heavysideZ) * cellSize - rayPos.z) * reciprocalRayZ;

        if (blockId == 0)
        {
            // how much rayDir should be scaled to intersect the nearest xyz grid planes respectively
            
            scaleX = Min((intRayPosX + heavysideX - rayPos.x + addedDist * signX) * reciprocalRayX, scaleX);
            scaleY = Min((intRayPosY + heavysideY - rayPos.y + addedDist * signY) * reciprocalRayY, scaleY);
            scaleZ = Min((intRayPosZ + heavysideZ - rayPos.z + addedDist * signZ) * reciprocalRayZ, scaleZ);
        }

        scale = Min(scaleX, Min(scaleY, scaleZ));

        Vec3D step = rayDir * scale;

        &rayPos += step;

        distance += manhattanRayLength * scale;
    }

    return false;
}

__global__ void RenderScene(GBuffer gBuffer, Vec3D playerPos, Matrix3X3 rotationMatrix, float zLength, WorldData worldData, const int screenCellW, const int screenCellH, const int frameCount, int globalBlockUpdate)
{
    const int screenCellX = (blockIdx.x * blockDim.x + threadIdx.x) * screenCellW;
    const int screenCellY = (blockIdx.y * blockDim.y + threadIdx.y) * screenCellH;

    uint32_t m_z = ((uint32_t(screenCellX) * 178262) % 311918 ^ (uint32_t(screenCellY) * 817376) % 211801 ^ (uint32_t(frameCount % 5137) * 701681) % 219321 + 13191) % 14893; // both of these are used for RNG
    uint32_t m_w = ((uint32_t(screenCellX) * 981723) % 419192 ^ (uint32_t(screenCellY) * 189107) % 110182 ^ (uint32_t(frameCount % 5137) * 310812) % 318291 + 37320) % 14893; // 14893

    RandState randState = { m_z, m_w };

    //RandState* randState = &randState1;//gBuffer.randStates + (screenCellY / screenCellH) * (SCREEN_W / screenCellW) + (screenCellX / screenCellW);

    int selectedPixelX = RandomUnsignedInt32(&randState) % 3 + screenCellX;
    int selectedPixelY = RandomUnsignedInt32(&randState) % 3 + screenCellY;

    int primaryIrradianceCachePos = -1;
    Vec3D rayPos;
    Vec3D rayDir;

    Block block;
    Voxel voxel;
    int voxelArrayPos;
    Matrix3X3 tangentMatrix;

    float distance;
#define MAX_DISTANCE 50.0f

    for (int y = screenCellY; y < screenCellY + screenCellH; ++y)
    {
        for (int x = screenCellX; x < screenCellX + screenCellW; ++x)
        {
            int currentIrradianceCachePos = -1;

            Vec3D currentRayPos = playerPos;
            Vec3D currentRayDir = rotationMatrix * Vec3D{ float(x - SCREEN_W / 2) / (SCREEN_W / 2), float(y - SCREEN_H / 2) / (SCREEN_W / 2), zLength };

            Block currentBlock;
            Voxel currentVoxel;
            int currentVoxelArrayPos;
            Matrix3X3 currentTangentMatrix;

            float currentDistance = 0;

            Vec3D specularColor = ZERO_VEC3D;
            float specularity = 0;

            bool firstHit = Raycast(currentRayPos, currentRayDir, MAX_DISTANCE, worldData.blocks, worldData.blockTypes, &currentRayPos, &currentBlock, &currentVoxel, &currentVoxelArrayPos, &currentTangentMatrix, &currentDistance);

            //Vec3D pixelAlbedo = ZERO_VEC3D;

            //int pixelNormalId = 0;

            if (firstHit)
            {
                //uchar4 voxelAlbedo = currentVoxel.albedo;
                //pixelAlbedo = { voxelAlbedo.x / 255.0f, voxelAlbedo.y / 255.0f, voxelAlbedo.z / 255.0f };

                int voxelFaceId = (currentTangentMatrix.jHat.x > 0) + 2 * (currentTangentMatrix.jHat.y < 0) + 3 * (currentTangentMatrix.jHat.y > 0) + 4 * (currentTangentMatrix.jHat.z < 0) + 5 * (currentTangentMatrix.jHat.z > 0); // hate this
                //pixelNormalId = voxelFaceId + 1;

                currentIrradianceCachePos = currentBlock.irradianceCacheOffset + worldData.cachePosLUT[currentVoxelArrayPos * VOXEL_FACES + voxelFaceId];

                if (x == selectedPixelX && y == selectedPixelY)
                {
                    primaryIrradianceCachePos = currentIrradianceCachePos;
                    rayPos = currentRayPos;
                    rayDir = currentRayDir;

                    block = currentBlock;
                    voxel = currentVoxel;
                    voxelArrayPos = currentVoxelArrayPos;
                    tangentMatrix = currentTangentMatrix;

                    distance = currentDistance;
                }

                if (currentVoxel.specularity > 0)
                {
                    float cosTerm = 1 + Dot(currentRayDir, currentTangentMatrix.jHat);

                    float cosTermSquared = cosTerm * cosTerm;

                    specularity = currentVoxel.specularity + (1 - currentVoxel.specularity) * cosTermSquared * cosTermSquared * cosTerm;

                    Vec3D specularVoxelColor = { currentVoxel.specularColor.x / 255.0f, currentVoxel.specularColor.y / 255.0f, currentVoxel.specularColor.z / 255.0f };

                    Vec3D reflectedRayDir = currentRayDir - 2 * Dot(currentRayDir, currentTangentMatrix.jHat) * currentTangentMatrix.jHat;

                    bool reflectedHit = Raycast(currentRayPos, reflectedRayDir, Max(MAX_DISTANCE - currentDistance, 0), worldData.blocks, worldData.blockTypes, &currentRayPos, &currentBlock, &currentVoxel, &currentVoxelArrayPos, &currentTangentMatrix, &currentDistance);

                    if (reflectedHit)
                    {
                        int voxelFaceId = (currentTangentMatrix.jHat.x > 0) + 2 * (currentTangentMatrix.jHat.y < 0) + 3 * (currentTangentMatrix.jHat.y > 0) + 4 * (currentTangentMatrix.jHat.z < 0) + 5 * (currentTangentMatrix.jHat.z > 0); // hate this

                        int irradianceCachePos = currentBlock.irradianceCacheOffset + worldData.cachePosLUT[currentVoxelArrayPos * VOXEL_FACES + voxelFaceId];

                        if (irradianceCachePos != -1)
                        {
                            specularColor = specularVoxelColor * worldData.irradianceCache[irradianceCachePos].irradiance * FastReciprocal(worldData.irradianceCache[irradianceCachePos].bounceCount);
                        }
                    }
                }
            }

            int pixelNumber = y * SCREEN_W + x;

            gBuffer.frameBuffer[pixelNumber] = ZERO_VEC3D;
            //gBuffer.albedoBuffer[pixelNumber] = pixelAlbedo;
            //gBuffer.tangentMatrixBuffer[pixelNumber] = currentTangentMatrix;
            //gBuffer.normalBuffer[pixelNumber] = pixelNormalId;
            //gBuffer.positionBuffer[pixelNumber] = currentRayPos;


            gBuffer.primaryIrradianceCachePositions[pixelNumber] = currentIrradianceCachePos;

            /*if (currentIrradianceCachePos != -1)
            {
                gBuffer.irradianceBuffer[pixelNumber] = worldData.irradianceCache[currentIrradianceCachePos];
            }*/
            
            gBuffer.updateVoxel[pixelNumber] = false;

            gBuffer.specularBuffer[pixelNumber] = specularColor;
            gBuffer.specularityBuffer[pixelNumber] = specularity;

            //int rotatedPixelNumber = x * SCREEN_H + y;

            //gBuffer.normalBufferRotated[rotatedPixelNumber] = pixelNormalId;
            //gBuffer.positionBufferRotated[rotatedPixelNumber] = currentRayPos;
        }
    }

    Vec3D outgoingLight = ZERO_VEC3D;
    Vec3D weight = { 1.0, 1.0, 1.0 };

    uchar4 voxelAlbedo = voxel.albedo;
    Vec3D diffuseTint = { voxelAlbedo.x / 255.0f, voxelAlbedo.y / 255.0f, voxelAlbedo.z / 255.0f };

    int primaryIrradianceBounceCount = 1;

    bool addContribution = false;

    bool pathTracing = true;

    if (primaryIrradianceCachePos == -1)
    {
        pathTracing = false;
    }
    else
    {
        primaryIrradianceBounceCount = worldData.irradianceCache[primaryIrradianceCachePos].bounceCount;
    }

    if (voxel.emittance > 0)
    {
        outgoingLight = diffuseTint * voxel.emittance;

        addContribution = true;

        pathTracing = false;
    }

    if (pathTracing)
    {
#define MAX_BOUNCES 2//2

        for (int i = 1; i <= MAX_BOUNCES; ++i)
        {
            Vec3D diffuseRayDir;
            Vec3D shadowRayDir;

            float rand1 = RandomFloat0To1(&randState);

            float randPhi = RandomFloat0To1(&randState) * TAU;

            float sinPhi;
            float cosPhi;

            FastSinCos(randPhi, &sinPhi, &cosPhi);

            // diffuse rayDir

            float sqrtRand1 = FastSqrt(rand1);

            diffuseRayDir = sqrtRand1 * cosPhi * tangentMatrix.iHat + FastSqrt(1 - rand1) * tangentMatrix.jHat + sqrtRand1 * sinPhi * tangentMatrix.kHat;

            // shadow rayDir

            float cosTheta = 1 - 2 * rand1;
            float sinTheta = FastSqrt(1 - cosTheta * cosTheta);

            int lightChunkIndex = (int(rayPos.z) / LIGHT_CHUNK_SIZE) * LIGHT_CHUNKS_H * LIGHT_CHUNKS_W + (int(rayPos.y) / LIGHT_CHUNK_SIZE) * LIGHT_CHUNKS_W + (int(rayPos.x) / LIGHT_CHUNK_SIZE);

            int lightsInLightChunk = worldData.lightsPerLightChunk[lightChunkIndex];

            // RIS with WRS

            Light lightInReservoir;
            float weightInReservoir;
            float cumulativeReservoirWeight = 0;

            for (int j = 0; j < PROPOSAL_LIGHT_SAMPLES; ++j)
            {
                int randLightInLightChunk = RandomUnsignedInt32(&randState) % uint32_t(lightsInLightChunk);

                int lightArrayIndex = lightChunkIndex * MAX_LIGHTS_PER_LIGHT_CHUNK + randLightInLightChunk;

                Light light = worldData.lights[lightArrayIndex];


                float emittance = light.emittance;

                Vec3D lightDirection = light.position - rayPos;

                float cosineTerm = Max(Dot(FastNormalize(lightDirection), tangentMatrix.jHat), 0.01);

                float distanceSquared = MagnitudeSquared(lightDirection);

                float weight = FastDivide(emittance * cosineTerm, distanceSquared);


                float randNum = RandomFloat0To1(&randState) * (cumulativeReservoirWeight + weight);

                if (randNum <= weight)
                {
                    lightInReservoir = light;
                    weightInReservoir = weight;
                }

                cumulativeReservoirWeight += weight;
            }

            float weightFromRIS = FastDivide((1.0 / PROPOSAL_LIGHT_SAMPLES) * cumulativeReservoirWeight, weightInReservoir);

            float lightRadius = lightInReservoir.radius;


            //Vec3D spherePoint = lightRadius * Vec3D{ sinTheta * cosPhi, cosTheta, sinTheta * sinPhi };

            Vec3D spherePoint = lightRadius * FastNormalize(Vec3D{ RandomFloat0To1(&randState) * 2 - 1, RandomFloat0To1(&randState) * 2 - 1, RandomFloat0To1(&randState) * 2 - 1 });

            shadowRayDir = FastNormalize(lightInReservoir.position + spherePoint - rayPos);

            float distance2 = MagnitudeSquared(lightInReservoir.position - rayPos);
            float lightPDF = 1 - FastSqrt(FastDivide(Max(distance2 - lightRadius * lightRadius, 0), distance2));

            float lightWeight = weightFromRIS * lightPDF * Max(Dot(shadowRayDir, tangentMatrix.jHat), 0) * lightsInLightChunk * 2;

            // new raycasts

            Vec3D directLight = SampleLight(rayPos, shadowRayDir, Max(MAX_DISTANCE - distance, 0), worldData.blocks, worldData.blockTypes) * lightWeight;

            &weight *= diffuseTint;

            if (i == MAX_BOUNCES)
            {
                float maxDirect = primaryIrradianceBounceCount * 0.01; //FastSqrt(primaryIrradianceBounceCount) * 0.05;
                
                directLight = { Min(directLight.x, maxDirect), Min(directLight.y, maxDirect), Min(directLight.z, maxDirect) }; // to avoid fireflies

                &outgoingLight += weight * directLight;

                break;
            }

            bool diffuseHit = Raycast(rayPos, diffuseRayDir, Max(MAX_DISTANCE - distance, 0), worldData.blocks, worldData.blockTypes, &rayPos, &block, &voxel, &voxelArrayPos, &tangentMatrix, &distance);

            if (diffuseHit)
            {
                int voxelFaceId = (tangentMatrix.jHat.x > 0) + 2 * (tangentMatrix.jHat.y < 0) + 3 * (tangentMatrix.jHat.y > 0) + 4 * (tangentMatrix.jHat.z < 0) + 5 * (tangentMatrix.jHat.z > 0); // hate this

                int irradianceCachePos = block.irradianceCacheOffset + worldData.cachePosLUT[voxelArrayPos * VOXEL_FACES + voxelFaceId];

                if (irradianceCachePos == -1)
                {
                    break;
                }

                int irradianceBounceCount = worldData.irradianceCache[irradianceCachePos].bounceCount;
                int currentUpdate = worldData.irradianceCache[irradianceCachePos].currentUpdate;

                #define HALF_WAY_POINT 300

                float indirectProbability = FastDivide(irradianceBounceCount, irradianceBounceCount + HALF_WAY_POINT);

                float randNum = RandomFloat0To1(&randState);

                if (randNum < indirectProbability && currentUpdate == globalBlockUpdate)
                {
                    Vec3D indirectLight = worldData.irradianceCache[irradianceCachePos].irradiance * FastReciprocal(irradianceBounceCount);

                    float maxIrradiance = primaryIrradianceBounceCount * 0.001; //FastSqrt(irradianceBounceCount) * 0.01;

                    indirectLight = { Min(indirectLight.x, maxIrradiance), Min(indirectLight.y, maxIrradiance), Min(indirectLight.z, maxIrradiance) }; // to avoid fireflies

                    &outgoingLight += weight * (directLight + indirectLight);

                    break;
                }

                &outgoingLight += weight * directLight;

                voxelAlbedo = voxel.albedo;
                diffuseTint = { voxelAlbedo.x / 255.0f, voxelAlbedo.y / 255.0f, voxelAlbedo.z / 255.0f };
            }
            else
            {
                break;
            }
        }

        addContribution = true;
    }

    // might want to do some kind of clamping of outgoingLight

    int pixelNumber = selectedPixelY * SCREEN_W + selectedPixelX;

    gBuffer.updateVoxel[pixelNumber] = addContribution;
    gBuffer.frameBuffer[pixelNumber] = outgoingLight;
}

__global__ void ForwardPathTracing(WorldData worldData, ReflectionPoint* reflectionPoints, RandState* randStates, Vec3D playerPos, int globalBlockUpdate)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;

    int irradianceCachePos = reflectionPoints[threadIndex].irradianceCachePos;

    if (irradianceCachePos != -1)
    {
        int currentUpdate = worldData.irradianceCache[irradianceCachePos].currentUpdate;

        if (currentUpdate != globalBlockUpdate)
        {
            worldData.irradianceCache[irradianceCachePos].irradiance = ZERO_VEC3D;
            worldData.irradianceCache[irradianceCachePos].bounceCount = 1;
            worldData.irradianceCache[irradianceCachePos].currentUpdate = globalBlockUpdate;
        }
    }


    RandState* randState = randStates + threadIndex;

    float continuationProbability = reflectionPoints[threadIndex].continuationProbability;

    if (RandomFloat0To1(randState) < continuationProbability)
    {
        Block block;
        Voxel voxel;
        int voxelArrayPos;
        Vec3D rayPos = reflectionPoints[threadIndex].pos;
        Matrix3X3 tangentMatrix = reflectionPoints[threadIndex].tangentMatrix;
        float distance = 0;


        float rand1 = RandomFloat0To1(randState);

        float randPhi = RandomFloat0To1(randState) * TAU;

        float sinPhi;
        float cosPhi;

        FastSinCos(randPhi, &sinPhi, &cosPhi);

        // diffuse rayDir

        float sqrtRand1 = FastSqrt(rand1);

        Vec3D diffuseRayDir = sqrtRand1 * cosPhi * tangentMatrix.iHat + FastSqrt(1 - rand1) * tangentMatrix.jHat + sqrtRand1 * sinPhi * tangentMatrix.kHat;

        // shadow rayDir

        float cosTheta = 1 - 2 * rand1;
        float sinTheta = FastSqrt(1 - cosTheta * cosTheta);

        int lightChunkIndex = (int(rayPos.z) / LIGHT_CHUNK_SIZE) * LIGHT_CHUNKS_H * LIGHT_CHUNKS_W + (int(rayPos.y) / LIGHT_CHUNK_SIZE) * LIGHT_CHUNKS_W + (int(rayPos.x) / LIGHT_CHUNK_SIZE);

        int lightsInLightChunk = worldData.lightsPerLightChunk[lightChunkIndex];

        // RIS with WRS

        Light lightInReservoir;
        float weightInReservoir;
        float cumulativeReservoirWeight = 0;

        for (int j = 0; j < PROPOSAL_LIGHT_SAMPLES; ++j)
        {
            int randLightInLightChunk = RandomUnsignedInt32(randState) % uint32_t(lightsInLightChunk);

            int lightArrayIndex = lightChunkIndex * MAX_LIGHTS_PER_LIGHT_CHUNK + randLightInLightChunk;

            Light light = worldData.lights[lightArrayIndex];


            float emittance = light.emittance;

            Vec3D lightDirection = light.position - rayPos;

            float cosineTerm = Max(Dot(FastNormalize(lightDirection), tangentMatrix.jHat), 0.01);

            float distanceSquared = MagnitudeSquared(lightDirection);

            float weight = FastDivide(emittance * cosineTerm, distanceSquared);


            float randNum = RandomFloat0To1(randState) * (cumulativeReservoirWeight + weight);

            if (randNum <= weight)
            {
                lightInReservoir = light;
                weightInReservoir = weight;
            }

            cumulativeReservoirWeight += weight;
        }

        float weightFromRIS = FastDivide((1.0 / PROPOSAL_LIGHT_SAMPLES) * cumulativeReservoirWeight, weightInReservoir);

        float lightRadius = lightInReservoir.radius;


        //Vec3D spherePoint = lightRadius * Vec3D{ sinTheta * cosPhi, cosTheta, sinTheta * sinPhi };

        Vec3D spherePoint = lightRadius * FastNormalize(Vec3D{ RandomFloat0To1(randState) * 2 - 1, RandomFloat0To1(randState) * 2 - 1, RandomFloat0To1(randState) * 2 - 1 });

        Vec3D shadowRayDir = FastNormalize(lightInReservoir.position + spherePoint - rayPos);

        float distance2 = MagnitudeSquared(lightInReservoir.position - rayPos);
        float lightPDF = 1 - FastSqrt(FastDivide(Max(distance2 - lightRadius * lightRadius, 0), distance2));

        float lightWeight = weightFromRIS * lightPDF * Max(Dot(shadowRayDir, tangentMatrix.jHat), 0) * lightsInLightChunk * 2;

        // new raycasts

        Vec3D directLight = SampleLight(rayPos, shadowRayDir, MAX_DISTANCE, worldData.blocks, worldData.blockTypes) * lightWeight;

        Vec3D indirectLight = ZERO_VEC3D;

        int previousIrradianceCachePos = reflectionPoints[threadIndex].irradianceCachePos;

        int previousIrradianceBounceCount = worldData.irradianceCache[previousIrradianceCachePos].bounceCount;

        Vec3D previousAlbedo = reflectionPoints[threadIndex].albedo;


        bool diffuseHit = Raycast(rayPos, diffuseRayDir, Max(MAX_DISTANCE - distance, 0), worldData.blocks, worldData.blockTypes, &rayPos, &block, &voxel, &voxelArrayPos, &tangentMatrix, &distance);

        if (diffuseHit)
        {
            int voxelFaceId = (tangentMatrix.jHat.x > 0) + 2 * (tangentMatrix.jHat.y < 0) + 3 * (tangentMatrix.jHat.y > 0) + 4 * (tangentMatrix.jHat.z < 0) + 5 * (tangentMatrix.jHat.z > 0); // hate this

            int irradianceCachePos = block.irradianceCacheOffset + worldData.cachePosLUT[voxelArrayPos * VOXEL_FACES + voxelFaceId];

            if (irradianceCachePos != -1)
            {
                indirectLight = worldData.irradianceCache[irradianceCachePos].irradiance * FastReciprocal(worldData.irradianceCache[irradianceCachePos].bounceCount);

                float maxIrradiance = previousIrradianceBounceCount * 0.01;

                indirectLight = { Min(indirectLight.x, maxIrradiance), Min(indirectLight.y, maxIrradiance), Min(indirectLight.z, maxIrradiance) };

                uchar4 voxelAlbedo = voxel.albedo;
                Vec3D diffuseTint = { voxelAlbedo.x / 255.0f, voxelAlbedo.y / 255.0f, voxelAlbedo.z / 255.0f };

                reflectionPoints[threadIndex].pos = rayPos;
                reflectionPoints[threadIndex].tangentMatrix = tangentMatrix;
                reflectionPoints[threadIndex].albedo = diffuseTint;
                reflectionPoints[threadIndex].irradianceCachePos = irradianceCachePos;
                reflectionPoints[threadIndex].continuationProbability *= 0.75;
            }
        }

        if (previousIrradianceBounceCount < MAX_IRRADIANCE_SAMPLES)
        {
            Vec3D outgoingLight = previousAlbedo * (directLight + indirectLight);

            atomicAdd(&(worldData.irradianceCache[previousIrradianceCachePos].irradiance.x), outgoingLight.x);
            atomicAdd(&(worldData.irradianceCache[previousIrradianceCachePos].irradiance.y), outgoingLight.y);
            atomicAdd(&(worldData.irradianceCache[previousIrradianceCachePos].irradiance.z), outgoingLight.z);
            atomicAdd(&(worldData.irradianceCache[previousIrradianceCachePos].bounceCount), 1);
        }
    }
    else
    {
        reflectionPoints[threadIndex].continuationProbability = 0;

        int lightChunkIndex = (int(playerPos.z) / LIGHT_CHUNK_SIZE) * LIGHT_CHUNKS_H * LIGHT_CHUNKS_W + (int(playerPos.y) / LIGHT_CHUNK_SIZE) * LIGHT_CHUNKS_W + (int(playerPos.x) / LIGHT_CHUNK_SIZE);

        int lightsInLightChunk = worldData.lightsPerLightChunk[lightChunkIndex];

        int randLightInLightChunk = RandomUnsignedInt32(randState) % uint32_t(lightsInLightChunk);

        int lightArrayIndex = lightChunkIndex * MAX_LIGHTS_PER_LIGHT_CHUNK + randLightInLightChunk;

        Light light = worldData.lights[lightArrayIndex];


        Vec3D randUnitVector = FastNormalize(Vec3D{ RandomFloat0To1(randState) * 2 - 1, RandomFloat0To1(randState) * 2 - 1, RandomFloat0To1(randState) * 2 - 1 });

        Vec3D randPointOnLightSource = randUnitVector * 0.49f + light.position;


        Block block;
        Voxel voxel;
        int voxelArrayPos;
        Matrix3X3 tangentMatrix;
        Vec3D rayPos;
        float distance = 0;


        bool hit = Raycast(randPointOnLightSource, randUnitVector, MAX_DISTANCE, worldData.blocks, worldData.blockTypes, &rayPos, &block, &voxel, &voxelArrayPos, &tangentMatrix, &distance);

        if (hit)
        {
            int voxelFaceId = (tangentMatrix.jHat.x > 0) + 2 * (tangentMatrix.jHat.y < 0) + 3 * (tangentMatrix.jHat.y > 0) + 4 * (tangentMatrix.jHat.z < 0) + 5 * (tangentMatrix.jHat.z > 0); // hate this

            int irradianceCachePos = block.irradianceCacheOffset + worldData.cachePosLUT[voxelArrayPos * VOXEL_FACES + voxelFaceId];

            if (irradianceCachePos != -1)
            {
                uchar4 voxelAlbedo = voxel.albedo;
                Vec3D diffuseTint = { voxelAlbedo.x / 255.0f, voxelAlbedo.y / 255.0f, voxelAlbedo.z / 255.0f };

                reflectionPoints[threadIndex].pos = rayPos;
                reflectionPoints[threadIndex].tangentMatrix = tangentMatrix;
                reflectionPoints[threadIndex].albedo = diffuseTint;
                reflectionPoints[threadIndex].irradianceCachePos = irradianceCachePos;
                reflectionPoints[threadIndex].continuationProbability = 1;
            }
        }
    }
}

__device__ void SetDistances(int threadCoord, int blockCoord, int* minDist1, int* minDist2)
{
    int dist1 = Max(Abs(threadCoord - blockCoord) - 1, 0) * BLOCK_RES;
    int dist2 = Max(threadCoord - 1, 0) * BLOCK_RES;

    if (blockCoord - threadCoord < 0)
    {
        dist2 = dist1;
        dist1 = Max(CHUNK_SIZE - threadCoord - 1, 0) * BLOCK_RES;
    }

    if (dist1 < *minDist1)
    {
        *minDist1 = dist1;
    }
    if (dist2 < *minDist2)
    {
        *minDist2 = dist2;
    }
}

__global__ void UpdateChunkDistanceFieldKernel(int chunkIndex, Block* blocks)
{
    int threadX = blockIdx.x * blockDim.x + threadIdx.x;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y;
    int threadZ = blockIdx.z * blockDim.z + threadIdx.z;

    int threadIndex = threadZ * CHUNK_SIZE * CHUNK_SIZE + threadY * CHUNK_SIZE + threadX;

    __shared__ bool blocksInChunk[CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE];

    int chunkOffset = chunkIndex * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

    int3 threadBlockPos;

    #define LOAD_INDEX(blockPos) (blockPos.z * blockDim.z + threadIdx.z) * CHUNK_SIZE * CHUNK_SIZE + (blockPos.y * blockDim.y + threadIdx.y) * CHUNK_SIZE + (blockPos.x * blockDim.x + threadIdx.x)

    threadBlockPos = { 0, 0, 0 };
    blocksInChunk[LOAD_INDEX(threadBlockPos)] = (blocks[chunkOffset + LOAD_INDEX(threadBlockPos)].id > 0);
    threadBlockPos = { 0, 0, 1 };
    blocksInChunk[LOAD_INDEX(threadBlockPos)] = (blocks[chunkOffset + LOAD_INDEX(threadBlockPos)].id > 0);
    threadBlockPos = { 0, 1, 0 };
    blocksInChunk[LOAD_INDEX(threadBlockPos)] = (blocks[chunkOffset + LOAD_INDEX(threadBlockPos)].id > 0);
    threadBlockPos = { 0, 1, 1 };
    blocksInChunk[LOAD_INDEX(threadBlockPos)] = (blocks[chunkOffset + LOAD_INDEX(threadBlockPos)].id > 0);
    threadBlockPos = { 1, 0, 0 };
    blocksInChunk[LOAD_INDEX(threadBlockPos)] = (blocks[chunkOffset + LOAD_INDEX(threadBlockPos)].id > 0);
    threadBlockPos = { 1, 0, 1 };
    blocksInChunk[LOAD_INDEX(threadBlockPos)] = (blocks[chunkOffset + LOAD_INDEX(threadBlockPos)].id > 0);
    threadBlockPos = { 1, 1, 0 };
    blocksInChunk[LOAD_INDEX(threadBlockPos)] = (blocks[chunkOffset + LOAD_INDEX(threadBlockPos)].id > 0);
    threadBlockPos = { 1, 1, 1 };
    blocksInChunk[LOAD_INDEX(threadBlockPos)] = (blocks[chunkOffset + LOAD_INDEX(threadBlockPos)].id > 0);

    __syncthreads();

    int minDistance = CHUNK_SIZE * 2;

    for (int z = 0; z < CHUNK_SIZE; ++z)
    {
        for (int y = 0; y < CHUNK_SIZE; ++y)
        {
            for (int x = 0; x < CHUNK_SIZE; ++x)
            {
                int blockIndex = z * CHUNK_SIZE * CHUNK_SIZE + y * CHUNK_SIZE + x;

                if (blocksInChunk[blockIndex])
                {
                    int distanceToBlock = Max(Abs(threadX - x), Max(Abs(threadY - y), Abs(threadZ - z))) - 1;

                    if (distanceToBlock < minDistance)
                    {
                        minDistance = distanceToBlock;
                    }
                }
            }
        }
    }

    blocks[chunkOffset + threadIndex].distance = minDistance;
}

__global__ void AddLightSourceKernel(Light lightSource, WorldData worldData)
{
    int3 chunkPos = { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z };
    int lightChunkIndex = chunkPos.z * LIGHT_CHUNKS_H * LIGHT_CHUNKS_W + chunkPos.y * LIGHT_CHUNKS_W + chunkPos.x;

    int lightsInLightChunk = worldData.lightsPerLightChunk[lightChunkIndex];

    if (lightsInLightChunk < MAX_LIGHTS_PER_LIGHT_CHUNK)
    {
        int lightArrayIndex = lightChunkIndex * MAX_LIGHTS_PER_LIGHT_CHUNK + lightsInLightChunk;

        worldData.lights[lightArrayIndex] = lightSource;

        ++worldData.lightsPerLightChunk[lightChunkIndex];
    }
}

__global__ void DeleteLightSourceKernel(Vec3D lightPos, WorldData worldData)
{
    int blockIndex = int(lightPos.z) * WORLD_H * WORLD_W + int(lightPos.y) * WORLD_W + int(lightPos.x);

    int3 chunkPos = { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z };
    int lightChunkIndex = chunkPos.z * LIGHT_CHUNKS_H * LIGHT_CHUNKS_W + chunkPos.y * LIGHT_CHUNKS_W + chunkPos.x;


    int lightsInLightChunk = worldData.lightsPerLightChunk[lightChunkIndex];


    for (int i = 0; i < MAX_LIGHTS_PER_LIGHT_CHUNK; ++i)
    {
        int lightArrayIndex = lightChunkIndex * MAX_LIGHTS_PER_LIGHT_CHUNK + i;

        Light currentLight = worldData.lights[lightArrayIndex];
        int currentLightBlockIndex = int(currentLight.position.z) * WORLD_H * WORLD_W + int(currentLight.position.y) * WORLD_W + int(currentLight.position.x);

        if (currentLightBlockIndex == blockIndex)
        {
            --lightsInLightChunk;
            worldData.lightsPerLightChunk[lightChunkIndex] = lightsInLightChunk;

            worldData.lights[lightArrayIndex] = worldData.lights[lightChunkIndex * MAX_LIGHTS_PER_LIGHT_CHUNK + lightsInLightChunk];

            break;
        }
    }
}

void AddLightSource(Light lightSource, WorldData worldData)
{
    dim3 grid(1, 1, 1);

    dim3 block(8, 8, 8);

    AddLightSourceKernel<<<grid, block>>>(lightSource, worldData);
}

void DeleteLightSource(Vec3D lightPos, WorldData worldData)
{
    dim3 grid(1, 1, 1);

    dim3 block(8, 8, 8);

    DeleteLightSourceKernel<<<grid, block>>>(lightPos, worldData);
}

void UpdateChunkDistanceField(int chunkIndex, Block* deviceBlocks)
{
    dim3 grid(2, 2, 2);

    dim3 block(8, 8, 8);

    UpdateChunkDistanceFieldKernel<<<grid, block>>>(chunkIndex, deviceBlocks);
}

__global__ void DrawSquare(cudaSurfaceObject_t screenCudaSurfaceObject, int startX, int startY)
{
    int threadX = blockIdx.x * blockDim.x + threadIdx.x + startX;
    int threadY = blockIdx.y * blockDim.y + threadIdx.y + startY;

    surf2Dwrite(make_uchar4(150, 150, 150, 255), screenCudaSurfaceObject, threadX * sizeof(uchar4), threadY);
}

void DrawCross(cudaSurfaceObject_t screenCudaSurfaceObject)
{
    dim3 grid(1, 1, 1); // 20, 45, 1

    dim3 block(8, 8, 1);

    DrawSquare<<<grid, block>>>(screenCudaSurfaceObject, SCREEN_W / 2 - 4, SCREEN_H / 2 - 4);
    DrawSquare<<<grid, block>>>(screenCudaSurfaceObject, SCREEN_W / 2 - 4 + 8, SCREEN_H / 2 - 4);
    DrawSquare<<<grid, block>>>(screenCudaSurfaceObject, SCREEN_W / 2 - 4 - 8, SCREEN_H / 2 - 4);
    DrawSquare<<<grid, block>>>(screenCudaSurfaceObject, SCREEN_W / 2 - 4, SCREEN_H / 2 - 4 + 8);
    DrawSquare<<<grid, block>>>(screenCudaSurfaceObject, SCREEN_W / 2 - 4, SCREEN_H / 2 - 4 - 8);
}

void Render(GBuffer gBuffer, WorldData worldData, ReflectionPoint* reflectionPoints, RandState* randStates)
{
    dim3 grid(20, 45, 1); // 20, 45, 1

    dim3 block(32, 8, 1); // 32, 8, 1

    const int screenCellW = 3;
    const int screenCellH = 3;

    cudaGraphicsMapResources(1, &screenCudaResource);
    
    cudaArray_t screenCudaArray;
    cudaGraphicsSubResourceGetMappedArray(&screenCudaArray, screenCudaResource, 0, 0);

    cudaResourceDesc screenCudaArrayResourceDesc;
    screenCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    screenCudaArrayResourceDesc.res.array.array = screenCudaArray;
        
    cudaSurfaceObject_t screenCudaSurfaceObject;
    cudaCreateSurfaceObject(&screenCudaSurfaceObject, &screenCudaArrayResourceDesc);

    Matrix3X3 rotationMatrix =
    {
        (player.orientation * Quaternion({ 0, { 1, 0, 0 } }) * Conjugate(player.orientation)).vecPart,
        (player.orientation * Quaternion({ 0, { 0, 1, 0 } }) * Conjugate(player.orientation)).vecPart,
        (player.orientation * Quaternion({ 0, { 0, 0, 1 } }) * Conjugate(player.orientation)).vecPart
    };

    float tanHalfFOV = tanf(player.FOV * 0.5);
    float zLength = 1.0 / tanHalfFOV;
    RenderScene<<<grid, block>>>(gBuffer, player.pos, rotationMatrix, zLength, worldData, screenCellW, screenCellH, frameCount, globalBlockUpdate);

    ForwardPathTracing<<<256, 256>>>(worldData, reflectionPoints, randStates, player.pos, globalBlockUpdate);

    // spacial filtering

    //AtrousFilter<<<grid, block>>>(gBuffer.frameBufferCopy, gBuffer.frameBuffer, gBuffer.normalBuffer, gBuffer.positionBuffer, 16, SCREEN_W, SCREEN_H, screenCellW, screenCellH);
    //AtrousFilter<<<grid, block>>>(gBuffer.frameBuffer, gBuffer.frameBufferCopy, gBuffer.normalBuffer, gBuffer.positionBuffer, 8, SCREEN_W, SCREEN_H, screenCellW, screenCellH, true);
    //AtrousFilter<<<grid, block>>>(gBuffer.frameBufferCopy, gBuffer.frameBuffer, gBuffer.normalBuffer, gBuffer.positionBuffer, 4, SCREEN_W, SCREEN_H, screenCellW, screenCellH);
    //AtrousFilter<<<grid, block>>>(gBuffer.frameBuffer, gBuffer.frameBufferCopy, gBuffer.normalBuffer, gBuffer.positionBuffer, 2, SCREEN_W, SCREEN_H, screenCellW, screenCellH);
    //AtrousFilter<<<grid, block>>>(gBuffer.frameBufferCopy, gBuffer.frameBuffer, gBuffer.normalBuffer, gBuffer.positionBuffer, 1, SCREEN_W, SCREEN_H, screenCellW, screenCellH, true);

    //dim3 rotatedGrid(45, 20, 1); // 20, 45, 1

    //dim3 rotatedBlock(8, 32, 1); // 32, 8, 1

    //AtrousFilter<<<rotatedGrid, rotatedBlock>>>(gBuffer.frameBuffer, gBuffer.frameBufferCopy, gBuffer.normalBufferRotated, gBuffer.positionBufferRotated, 16, SCREEN_H, SCREEN_W, screenCellH, screenCellW);
    //AtrousFilter<<<rotatedGrid, rotatedBlock>>>(gBuffer.frameBufferCopy, gBuffer.frameBuffer, gBuffer.normalBufferRotated, gBuffer.positionBufferRotated, 8, SCREEN_H, SCREEN_W, screenCellH, screenCellW, true);
    //AtrousFilter<<<rotatedGrid, rotatedBlock>>>(gBuffer.frameBuffer, gBuffer.frameBufferCopy, gBuffer.normalBufferRotated, gBuffer.positionBufferRotated, 4, SCREEN_H, SCREEN_W, screenCellH, screenCellW);
    //AtrousFilter<<<rotatedGrid, rotatedBlock>>>(gBuffer.frameBufferCopy, gBuffer.frameBuffer, gBuffer.normalBufferRotated, gBuffer.positionBufferRotated, 2, SCREEN_H, SCREEN_W, screenCellH, screenCellW);
    //AtrousFilter<<<rotatedGrid, rotatedBlock>>>(gBuffer.frameBuffer, gBuffer.frameBufferCopy, gBuffer.normalBufferRotated, gBuffer.positionBufferRotated, 1, SCREEN_H, SCREEN_W, screenCellH, screenCellW, true);

    accumulatedFrameCount += (accumulatedFrameCount < 100000);
    ++frameCount;
    frameCount = frameCount % 100000;

    //DeamplifyVoxelColors<<<grid, block>>>(gBuffer, worldData, screenCellW, screenCellH);
    DeamplifyVoxels<<<grid, block>>>(gBuffer, worldData, screenCellW, screenCellH, globalBlockUpdate);
    AccumulateVoxels<<<grid, block>>>(gBuffer, worldData, screenCellW, screenCellH);

    DrawFrame<<<grid, block>>>(gBuffer, worldData, player.pos, rotationMatrix, tanHalfFOV, screenCudaSurfaceObject, screenCellW, screenCellH);

    DrawCross(screenCudaSurfaceObject);
        
    cudaDestroySurfaceObject(screenCudaSurfaceObject);
    
    cudaGraphicsUnmapResources(1, &screenCudaResource);

    cudaStreamSynchronize(0);

    glBindTexture(GL_TEXTURE_2D, screenGLTexture);
    
    glBegin(GL_QUADS);
    
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
    
    glEnd();
    
    glBindTexture(GL_TEXTURE_2D, 0);

    glFinish();
}

bool KeyPressed(GLFWwindow* window, int key, bool* keyRepeat)
{
    int state = glfwGetKey(window, key);

    if (state == GLFW_RELEASE)
    {
        *keyRepeat = false;
        return false;
    }
    else if (state == GLFW_PRESS && *keyRepeat == false)
    {
        *keyRepeat = true;
        return true;
    }
    else
    {
        return false;
    }
}

void Input(GLFWwindow* window)
{
#define MAX_FRAME_COUNT 1

    if (glfwGetKey(window, GLFW_KEY_W))
    {
        Quaternion q_newDirection = player.orientation * Quaternion({ 0, { 0, 0, 1 } }) * Conjugate(player.orientation);

        Normalize(&q_newDirection.vecPart);
        Scale(&q_newDirection.vecPart, player.MovementSpeed(elapsedTime));

        &player.pos += q_newDirection.vecPart;
    }
    if (glfwGetKey(window, GLFW_KEY_A))
    {
        Quaternion q_newDirection = player.orientation * Quaternion({ 0, { -1, 0, 0 } }) * Conjugate(player.orientation);

        Normalize(&q_newDirection.vecPart);
        Scale(&q_newDirection.vecPart, player.MovementSpeed(elapsedTime));

        &player.pos += q_newDirection.vecPart;
    }
    if (glfwGetKey(window, GLFW_KEY_S))
    {
        Quaternion q_newDirection = player.orientation * Quaternion({ 0, { 0, 0, -1 } }) * Conjugate(player.orientation);

        Normalize(&q_newDirection.vecPart);
        Scale(&q_newDirection.vecPart, player.MovementSpeed(elapsedTime));

        &player.pos += q_newDirection.vecPart;
    }
    if (glfwGetKey(window, GLFW_KEY_D))
    {
        Quaternion q_newDirection = player.orientation * Quaternion({ 0, { 1, 0, 0 } }) * Conjugate(player.orientation);

        Normalize(&q_newDirection.vecPart);
        Scale(&q_newDirection.vecPart, player.MovementSpeed(elapsedTime));

        &player.pos += q_newDirection.vecPart;
    }

    if (glfwGetKey(window, GLFW_KEY_SPACE))
    {
        player.pos.y += player.MovementSpeed(elapsedTime);
    }

    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT))
    {
        player.pos.y -= player.MovementSpeed(elapsedTime);
    }

    if (glfwGetKey(window, GLFW_KEY_RIGHT))
    {
        Normalize(&player.orientation);

        Quaternion q_newRotationAxis = Conjugate(player.orientation) * Quaternion({ 0, { 0, 1, 0 } }) * player.orientation;

        Quaternion rotationQuaternion = CreateRotQuat(q_newRotationAxis.vecPart, player.RotationSpeed(elapsedTime));

        player.orientation = player.orientation * rotationQuaternion;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT))
    {
        Normalize(&player.orientation);

        Quaternion q_newRotationAxis = Conjugate(player.orientation) * Quaternion({ 0, { 0, 1, 0 } }) * player.orientation;

        Quaternion rotationQuaternion = CreateRotQuat(q_newRotationAxis.vecPart, -player.RotationSpeed(elapsedTime));

        player.orientation = player.orientation * rotationQuaternion;
    }
    if (glfwGetKey(window, GLFW_KEY_UP))
    {
        Normalize(&player.orientation);

        Quaternion rotationQuaternion = CreateRotQuat({ 1, 0, 0 }, -player.RotationSpeed(elapsedTime));

        player.orientation = player.orientation * rotationQuaternion;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN))
    {
        Normalize(&player.orientation);

        Quaternion rotationQuaternion = CreateRotQuat({ 1, 0, 0 }, player.RotationSpeed(elapsedTime));

        player.orientation = player.orientation * rotationQuaternion;
    }

    player.inputFlag = IDLE;

    if (KeyPressed(window, GLFW_KEY_Q, &Q_REPEAT))
    {
        player.inputFlag = PLACE_BLOCK;
    }
    if (KeyPressed(window, GLFW_KEY_E, &E_REPEAT))
    {
        player.inputFlag = REMOVE_BLOCK;
    }

    if (glfwGetKey(window, GLFW_KEY_1))
    {
        inventoryId = 1;
    }
    else if (glfwGetKey(window, GLFW_KEY_2))
    {
        inventoryId = 2;
    }
    else if (glfwGetKey(window, GLFW_KEY_3))
    {
        inventoryId = 3;
    }
    else if (glfwGetKey(window, GLFW_KEY_4))
    {
        inventoryId = 4;
    }
    else if (glfwGetKey(window, GLFW_KEY_5))
    {
        inventoryId = 5;
    }
    else if (glfwGetKey(window, GLFW_KEY_6))
    {
        inventoryId = 6;
    }
    else if (glfwGetKey(window, GLFW_KEY_7))
    {
        inventoryId = 7;
    }
    else if (glfwGetKey(window, GLFW_KEY_8))
    {
        inventoryId = 8;
    }
    else if (glfwGetKey(window, GLFW_KEY_9))
    {
        inventoryId = 9;
    }
    else if (glfwGetKey(window, GLFW_KEY_0))
    {
        inventoryId = 10;
    }
}

void Simulate(Block* hostBlocks, WorldData worldData)
{
    if (player.inputFlag == IDLE)
    {
        return;
    }
    
    Vec3D rayPos = player.pos;
    Vec3D rayDir = (player.orientation * Quaternion({ 0, { 0, 0, 1 } }) * Conjugate(player.orientation)).vecPart; // rotate it

    float heavisideX = (rayDir.x >= 0) * 1.002 - 0.001; // accounting for dumb floating point errors
    float heavisideY = (rayDir.y >= 0) * 1.002 - 0.001;
    float heavisideZ = (rayDir.z >= 0) * 1.002 - 0.001;

    float distance = 0;

    #define MAX_INTERACT_DISTANCE 5.0

    while (distance < MAX_INTERACT_DISTANCE)
    {
        float scaleX = (int(rayPos.x) + heavisideX - rayPos.x) / rayDir.x;
        float scaleY = (int(rayPos.y) + heavisideY - rayPos.y) / rayDir.y;
        float scaleZ = (int(rayPos.z) + heavisideZ - rayPos.z) / rayDir.z;

        float scale = Min(scaleX, Min(scaleY, scaleZ));

        Vec3D step = rayDir * scale;

        Vec3D nextRayPos = rayPos + step;

        if (nextRayPos.x < 0 || nextRayPos.x >= WORLD_W || nextRayPos.y < 0 || nextRayPos.y >= WORLD_H || nextRayPos.z < 0 || nextRayPos.z >= WORLD_L)
        {
            break;
        }

        int chunkIndex = (int(nextRayPos.z) / CHUNK_SIZE) * (WORLD_H / CHUNK_SIZE) * (WORLD_W / CHUNK_SIZE) + (int(nextRayPos.y) / CHUNK_SIZE) * (WORLD_W / CHUNK_SIZE) + (int(nextRayPos.x) / CHUNK_SIZE);

        int arrayPos = chunkIndex * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + (int(nextRayPos.z) % CHUNK_SIZE) * CHUNK_SIZE * CHUNK_SIZE + (int(nextRayPos.y) % CHUNK_SIZE) * CHUNK_SIZE + (int(nextRayPos.x) % CHUNK_SIZE);

        Block block = hostBlocks[arrayPos];
        
        if (block.id > 0)
        {
            if (player.inputFlag == REMOVE_BLOCK)
            {
                if (block.id > 5)
                {
                    DeleteLightSource(nextRayPos, worldData);
                }

                hostBlocks[arrayPos].id = 0;
                cudaMemcpy(worldData.blocks + arrayPos, hostBlocks + arrayPos, sizeof(Block), cudaMemcpyHostToDevice);
                UpdateChunkDistanceField(chunkIndex, worldData.blocks);

                ++globalBlockUpdate;
            }
            else if (player.inputFlag == PLACE_BLOCK)
            {
                int previousChunkIndex = (int(rayPos.z) / CHUNK_SIZE) * (WORLD_H / CHUNK_SIZE) * (WORLD_W / CHUNK_SIZE) + (int(rayPos.y) / CHUNK_SIZE) * (WORLD_W / CHUNK_SIZE) + (int(rayPos.x) / CHUNK_SIZE);
                int previousArrayPos = previousChunkIndex * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + (int(rayPos.z) % CHUNK_SIZE) * CHUNK_SIZE * CHUNK_SIZE + (int(rayPos.y) % CHUNK_SIZE) * CHUNK_SIZE + (int(rayPos.x) % CHUNK_SIZE);

                if (inventoryId > 5)
                {
                    Light lightToAdd = { { int(rayPos.x) + 0.5f, int(rayPos.y) + 0.5f, int(rayPos.z) + 0.5f }, 0.35, 1.0 };

                    AddLightSource(lightToAdd, worldData);
                }

                hostBlocks[previousArrayPos].id = inventoryId;
                hostBlocks[previousArrayPos].irradianceCacheOffset = cachedFaces;
                cachedFaces += cacheSizes[inventoryId];

                cudaMemcpy(worldData.blocks + previousArrayPos, hostBlocks + previousArrayPos, sizeof(Block), cudaMemcpyHostToDevice);
                UpdateChunkDistanceField(previousChunkIndex, worldData.blocks);

                ++globalBlockUpdate;
            }

            break;
        }

        rayPos = nextRayPos;

        distance += Magnitude(step);
    }
    
}

void PutBlock(Block* blocks, int x, int y, int z, int id)
{
    int chunkIndex = (z / CHUNK_SIZE) * (WORLD_H / CHUNK_SIZE) * (WORLD_W / CHUNK_SIZE) + (y / CHUNK_SIZE) * (WORLD_W / CHUNK_SIZE) + (x / CHUNK_SIZE);

    blocks[chunkIndex * CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE + (z % CHUNK_SIZE) * CHUNK_SIZE * CHUNK_SIZE + (y % CHUNK_SIZE) * CHUNK_SIZE + (x % CHUNK_SIZE)].id = id;
}

bool LocateExitInBlock(int i, int3 voxelPos, Voxel* blockTypes, bool* visitedVoxels)
{
    auto GetPosInArray = [](int i, int3 pos)
    {
        return (i * BLOCK_RES * BLOCK_RES * BLOCK_RES) + pos.z * BLOCK_RES * BLOCK_RES + pos.y * BLOCK_RES + pos.x;
    };

    auto CheckIfFull = [](int pos, Voxel* blockTypes)
    {
        return (blockTypes[pos].albedo.x != 0 || blockTypes[pos].albedo.y != 0 || blockTypes[pos].albedo.z != 0 || blockTypes[pos].albedo.w != 0);
    };

    auto CheckIfVoxelIsInsideBlock = [](int3 pos)
    {
        return (pos.x >= 0 && pos.x < BLOCK_RES && pos.y >= 0 && pos.y < BLOCK_RES && pos.z >= 0 && pos.z < BLOCK_RES);
    };

    visitedVoxels[voxelPos.z * BLOCK_RES * BLOCK_RES + voxelPos.y * BLOCK_RES + voxelPos.x] = true;

    for (int j = 0; j < VOXEL_FACES; ++j)
    {
        int3 neighbour;

        switch (j)
        {
        case 0:
            neighbour = { voxelPos.x - 1, voxelPos.y, voxelPos.z };
            break;
        case 1:
            neighbour = { voxelPos.x + 1, voxelPos.y, voxelPos.z };
            break;
        case 2:
            neighbour = { voxelPos.x, voxelPos.y - 1, voxelPos.z };
            break;
        case 3:
            neighbour = { voxelPos.x, voxelPos.y + 1, voxelPos.z };
            break;
        case 4:
            neighbour = { voxelPos.x, voxelPos.y, voxelPos.z - 1 };
            break;
        case 5:
            neighbour = { voxelPos.x, voxelPos.y, voxelPos.z + 1 };
            break;
        }

        if (CheckIfVoxelIsInsideBlock(neighbour) == false)
        {
            return true;
        }
        else if (CheckIfFull(GetPosInArray(i, neighbour), blockTypes) == false && visitedVoxels[neighbour.z * BLOCK_RES * BLOCK_RES + neighbour.y * BLOCK_RES + neighbour.x] == false)
        {
            if (LocateExitInBlock(i, neighbour, blockTypes, visitedVoxels))
            {
                return true;
            }
        }
    }

    return false;
}

int main(void)
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Device name: " << prop.name << '\n';
    std::cout << "Memory Clock Rate (KHz): " << prop.memoryClockRate << '\n';
    std::cout << "Memory Bus Width (bits): " << prop.memoryBusWidth << '\n';
    std::cout << "Peak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << '\n';
    std::cout << "Warp Size in Threads: " << prop.warpSize << '\n';
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << '\n';
    std::cout << "Max Threads per Block-dimension: " << prop.maxThreadsDim << '\n';
    std::cout << "Max Blocks per Grid-dimension: " << prop.maxGridSize << '\n';
    std::cout << "Max Threads per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << '\n';
    std::cout << "Max Blocks per Multiprocessor: " << prop.maxBlocksPerMultiProcessor << '\n';
    std::cout << "Numer of Multiprocessors: " << prop.multiProcessorCount << '\n';
    std::cout << "Shared Memory per Block (bytes): " << prop.sharedMemPerBlock << '\n';
    std::cout << "Shared Memory per Multiprocessor (bytes): " << prop.sharedMemPerMultiprocessor << '\n';
    std::cout << "Device Can Map Host Memory: " << prop.canMapHostMemory << '\n';
    std::cout << "L2 Cache Size (bytes): " << prop.l2CacheSize << '\n';
    std::cout << "Voxel Size: " << sizeof(Voxel) << '\n';

    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(SCREEN_W, SCREEN_H, "Voxel Engine", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    // copied initialization code
    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &screenGLTexture);

    glBindTexture(GL_TEXTURE_2D, screenGLTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCREEN_W, SCREEN_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);

    cudaGraphicsGLRegisterImage(&screenCudaResource, screenGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
    //end of initialization code

    Vec3D* frameBuffer;
    cudaMalloc(&frameBuffer, SCREEN_W * SCREEN_H * sizeof(Vec3D));
    Vec3D* frameBufferCopy;
    cudaMalloc(&frameBufferCopy, SCREEN_W * SCREEN_H * sizeof(Vec3D));
    
    Vec3D* albedoBuffer;
    cudaMalloc(&albedoBuffer, SCREEN_W * SCREEN_H * sizeof(Vec3D));

    /*uint8_t* normalBuffer;
    cudaMalloc(&normalBuffer, SCREEN_W * SCREEN_H * sizeof(uint8_t));
    uint8_t* normalBufferRotated;
    cudaMalloc(&normalBufferRotated, SCREEN_W * SCREEN_H * sizeof(uint8_t));*/

    //Matrix3X3* tangentMatrixBuffer;
    //cudaMalloc(&tangentMatrixBuffer, SCREEN_W * SCREEN_H * sizeof(Matrix3X3));

    /*Vec3D* positionBuffer;
    cudaMalloc(&positionBuffer, SCREEN_W * SCREEN_H * sizeof(Vec3D));
    Vec3D* positionBufferRotated;
    cudaMalloc(&positionBufferRotated, SCREEN_W * SCREEN_H * sizeof(Vec3D));*/

    //std::mt19937 randomEngine(seedEngine());

    /*std::mt19937 randomEngine(seedEngine());

    RandState* hostRandScreenStates = (RandState*)malloc((SCREEN_W / 3) * (SCREEN_H / 3) * sizeof(RandState));

    for (int i = 0; i < (SCREEN_W  / 3) * (SCREEN_H / 3); ++i)
    {
        hostRandScreenStates[i].m_w = randomEngine() % 65535;
        hostRandScreenStates[i].m_z = randomEngine() % 65535;

        //std::cout << hostRandScreenStates[i].m_w << '\n';
    }

    RandState* deviceRandScreenStates;
    cudaMalloc(&deviceRandScreenStates, (SCREEN_W / 3) * (SCREEN_H / 3) * sizeof(RandState));
    cudaMemcpy(deviceRandScreenStates, hostRandScreenStates, (SCREEN_W / 3) * (SCREEN_H / 3) * sizeof(RandState), cudaMemcpyHostToDevice);*/

    int* primaryIrradianceCachePositions;
    cudaMalloc(&primaryIrradianceCachePositions, SCREEN_W * SCREEN_H * sizeof(int));

    //IrradianceCache* irradianceBuffer;
    //cudaMalloc(&irradianceBuffer, SCREEN_W * SCREEN_H * sizeof(IrradianceCache));

    bool* updateVoxel;
    cudaMalloc(&updateVoxel, SCREEN_W * SCREEN_H * sizeof(bool));

    Vec3D* specularBuffer;
    cudaMalloc(&specularBuffer, SCREEN_W * SCREEN_H * sizeof(Vec3D));
    float* specularityBuffer;
    cudaMalloc(&specularityBuffer, SCREEN_W * SCREEN_H * sizeof(float));

    //GBuffer gBuffer = { frameBuffer, frameBufferCopy, albedoBuffer, normalBuffer, normalBufferRotated, tangentMatrixBuffer, positionBuffer, positionBufferRotated, primaryIrradianceCachePositions, irradianceBuffer, updateVoxel, specularBuffer, specularityBuffer };

    //GBuffer gBuffer = { frameBuffer, frameBufferCopy, albedoBuffer, deviceRandScreenStates, primaryIrradianceCachePositions, irradianceBuffer, updateVoxel, specularBuffer, specularityBuffer };

    GBuffer gBuffer = { frameBuffer, frameBufferCopy, albedoBuffer, primaryIrradianceCachePositions, updateVoxel, specularBuffer, specularityBuffer };

    int blockBytes = WORLD_W * WORLD_H * WORLD_L * sizeof(Block);
    Block* hostBlocks = (Block*)malloc(blockBytes);

    for (int i = 0; i < WORLD_W * WORLD_L * WORLD_H; ++i)
    {
        hostBlocks[i].id = 0;
        hostBlocks[i].distance = 0;
    }

    for (int z = 20; z <= 40; ++z)
    {
        for (int y = 20; y <= 25; ++y)
        {
            for (int x = 20; x <= 40; ++x)
            {
                if (rand() % 2 == 0)
                {
                    //PutBlock(hostBlocks, x, y, z, 2);
                }
            }
        }
    }
    
    for (int z = 20; z <= 40; ++z)
    {
        for (int x = 20; x <= 40; ++x)
        {
            PutBlock(hostBlocks, x, 26, z, 3);

            PutBlock(hostBlocks, x, 9, z, 5);

            if (rand() % 16 == 0)
            {
                //PutBlock(hostBlocks, x, 27, z, 5);
            }
        }
    }

    PutBlock(hostBlocks, 30, 27, 30, 6);
    PutBlock(hostBlocks, 35, 27, 39, 6);
    PutBlock(hostBlocks, 39, 27, 23, 6);
    PutBlock(hostBlocks, 20, 10, 40, 7);

    //PutBlock(hostBlocks, 25, 10, 35, 11);

    for (int y = 27; y < 30; ++y)
    {
        for (int x = 36; x <= 40; ++x)
        {
            PutBlock(hostBlocks, x, y, 20, 2);
            PutBlock(hostBlocks, x, y, 24, 2);
        }

        for (int z = 20; z <= 24; ++z)
        {
            PutBlock(hostBlocks, 36, y, z, 2);
            PutBlock(hostBlocks, 40, y, z, 2);
        }
    }

    for (int z = 20; z <= 24; ++z)
    {
        for (int x = 36; x <= 40; ++x)
        {
            PutBlock(hostBlocks, x, 30, z, 2);
        }
    }

    PutBlock(hostBlocks, 36, 27, 22, 0);
    PutBlock(hostBlocks, 36, 28, 22, 0);

    for (int i = 0; i < 20; ++i)
    {
        int randX = rand() % 20 + 20;
        int randZ = rand() % 20 + 20;

        for (int x = randX - 1; x <= randX + 1; ++x)
        {
            for (int z = randZ - 1; z <= randZ + 1; ++z)
            {
                PutBlock(hostBlocks, x, 31, z, 1);
                PutBlock(hostBlocks, x, 32, z, 1);
            }
        }

        PutBlock(hostBlocks, randX, 33, randZ, 1);

        for (int y = 27; y <= 31; ++y)
        {
            PutBlock(hostBlocks, randX, y, randZ, 4);
        }
    }

    PutBlock(hostBlocks, 30, 31, 31, 4);
    PutBlock(hostBlocks, 31, 31, 31, 4);

    for (int x = 31; x <= 35; ++x)
    {
        for (int z = 31; z <= 35; ++z)
        {
            PutBlock(hostBlocks, x, 27, z, 3);
        }
    }

    PutBlock(hostBlocks, 28, 27, 26, 10);
    PutBlock(hostBlocks, 38, 27, 38, 10);
    PutBlock(hostBlocks, 25, 27, 30, 10);
    PutBlock(hostBlocks, 37, 27, 27, 10);


    int blockTypeBytes = BLOCK_TYPES * BLOCK_RES * BLOCK_RES * BLOCK_RES * sizeof(Voxel);

    Voxel* hostBlockTypes = (Voxel*)malloc(blockTypeBytes);

    for (int i = 0; i < BLOCK_RES * BLOCK_RES * BLOCK_RES; ++i)
    {
        hostBlockTypes[i].albedo = make_uchar4(0, 0, 0, 0);
    }

    for (int i = BLOCK_RES * BLOCK_RES * BLOCK_RES; i < BLOCK_RES * BLOCK_RES * BLOCK_RES * 2; ++i)
    {
        hostBlockTypes[i].albedo = make_uchar4(0, 0, 0, 0);

        if (rand() % 6 == 0)
        {
            hostBlockTypes[i].albedo = make_uchar4(rand() % 75, rand() % 155 + 100, rand() % 25 + 25, 255);
            hostBlockTypes[i].emittance = 0;
            hostBlockTypes[i].specularity = 0;
            hostBlockTypes[i].specularColor = make_uchar4(0, 0, 0, 0);
        }
    }

    for (int i = BLOCK_RES * BLOCK_RES * BLOCK_RES * 2; i < BLOCK_RES * BLOCK_RES * BLOCK_RES * 3; ++i)
    {
        hostBlockTypes[i].albedo = make_uchar4(0, 0, 0, 0);

        if (rand() % 4 < 3 )
        {
            int red = rand() % 25 + 100;
            hostBlockTypes[i].albedo = make_uchar4(red, red - 40 + rand() % 15, 0, 255);
            hostBlockTypes[i].emittance = 0;
            hostBlockTypes[i].specularity = 0;
            hostBlockTypes[i].specularColor = make_uchar4(0, 0, 0, 0);
        }
    }

    for (int i = BLOCK_RES * BLOCK_RES * BLOCK_RES * 3; i < BLOCK_RES * BLOCK_RES * BLOCK_RES * 4; ++i)
    {
        hostBlockTypes[i].albedo = make_uchar4(0, 0, 0, 0);

        if (i % (BLOCK_RES * BLOCK_RES) >= BLOCK_RES * 13)
        {
            if (rand() % 16 < 14)
            {
                hostBlockTypes[i].albedo = make_uchar4(rand() % 100, rand() % 100 + 100, rand() % 25 + 25, 255);
                hostBlockTypes[i].emittance = 0;
                hostBlockTypes[i].specularity = 0;
                hostBlockTypes[i].specularColor = make_uchar4(0, 0, 0, 0);
            }
        }
        else
        {
            if (rand() % 4 < 3)
            {
                int red = rand() % 25 + 100;
                hostBlockTypes[i].albedo = make_uchar4(red, red - 40 + rand() % 15, 0, 255);
                hostBlockTypes[i].emittance = 0;
                hostBlockTypes[i].specularity = 0;
                hostBlockTypes[i].specularColor = make_uchar4(0, 0, 0, 0);
            }
        }
    }

    for (int i = BLOCK_RES * BLOCK_RES * BLOCK_RES * 4; i < BLOCK_RES * BLOCK_RES * BLOCK_RES * 5; ++i)
    {
        hostBlockTypes[i].albedo = make_uchar4(0, 0, 0, 0);

        float x = (i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 4) % BLOCK_RES + 0.5;
        float z = (i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 4) / (BLOCK_RES * BLOCK_RES) + 0.5;

        float dist = sqrtf((x - BLOCK_RES / 2) * (x - BLOCK_RES / 2) + (z - BLOCK_RES / 2) * (z - BLOCK_RES / 2));

        if (dist < (BLOCK_RES / 2) && rand() % 7 < 6)
        {
            if (((i % (BLOCK_RES * BLOCK_RES)) / BLOCK_RES) % 2 == 0)
            {
                int red = rand() % 35 + 125;
                hostBlockTypes[i].albedo = make_uchar4(red, red - 75 + rand() % 20, 0, 255);
                hostBlockTypes[i].emittance = 0;
                hostBlockTypes[i].specularity = 0;
                hostBlockTypes[i].specularColor = make_uchar4(0, 0, 0, 0);
            }
            else
            {
                int red = rand() % 25 + 75;
                hostBlockTypes[i].albedo = make_uchar4(red, red - 50 + rand() % 15, 0, 255);
                hostBlockTypes[i].emittance = 0;
                hostBlockTypes[i].specularity = 0;
                hostBlockTypes[i].specularColor = make_uchar4(0, 0, 0, 0);
            }
        }
    }

    for (int i = BLOCK_RES * BLOCK_RES * BLOCK_RES * 5; i < BLOCK_RES * BLOCK_RES * BLOCK_RES * 6; ++i)
    {
        hostBlockTypes[i].albedo = make_uchar4(0, 0, 0, 0);

        if (rand() % 5 < 4)
        {
            int brightness = rand() % 75 + 50;
            hostBlockTypes[i].albedo = make_uchar4(brightness, brightness, brightness + 20, 255);
            hostBlockTypes[i].emittance = 0;
            hostBlockTypes[i].specularity = 0;
            hostBlockTypes[i].specularColor = make_uchar4(0, 0, 0, 0);
        }
    }

    float multiplier = 3;

    for (int i = BLOCK_RES * BLOCK_RES * BLOCK_RES * 6; i < BLOCK_RES * BLOCK_RES * BLOCK_RES * 7; ++i)
    {
        hostBlockTypes[i].albedo = make_uchar4(0, 0, 0, 0);

        float x = (i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 6) % BLOCK_RES + 0.5;
        float y = ((i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 6) / BLOCK_RES) % BLOCK_RES + 0.5;
        float z = (i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 6) / (BLOCK_RES * BLOCK_RES) + 0.5;

        float dist = sqrtf((x - BLOCK_RES / 2) * (x - BLOCK_RES / 2) + (y - BLOCK_RES / 2) * (y - BLOCK_RES / 2) + (z - BLOCK_RES / 2) * (z - BLOCK_RES / 2));

        if (dist < 6)
        {
            //hostBlockTypes[i].albedo = make_uchar4(rand() % 200 + 55, rand() % 50, rand() % 125 + 50, 255);
            hostBlockTypes[i].albedo = make_uchar4(255, 255, 255, 255); //make_uchar4(255, 12, 167, 255);
            hostBlockTypes[i].emittance = 56.0f * multiplier;
            hostBlockTypes[i].specularity = 0;
            hostBlockTypes[i].specularColor = make_uchar4(0, 0, 0, 0);
        }
    }

    for (int i = BLOCK_RES * BLOCK_RES * BLOCK_RES * 7; i < BLOCK_RES * BLOCK_RES * BLOCK_RES * 8; ++i)
    {
        hostBlockTypes[i].albedo = make_uchar4(0, 0, 0, 0);

        float x = (i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 7) % BLOCK_RES + 0.5;
        float y = ((i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 7) / BLOCK_RES) % BLOCK_RES + 0.5;
        float z = (i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 7) / (BLOCK_RES * BLOCK_RES) + 0.5;

        float dist = sqrtf((x - BLOCK_RES / 2) * (x - BLOCK_RES / 2) + (y - BLOCK_RES / 2) * (y - BLOCK_RES / 2) + (z - BLOCK_RES / 2) * (z - BLOCK_RES / 2));

        if (dist < 6)
        {
            //hostBlockTypes[i].albedo = make_uchar4(rand() % 200 + 55, rand() % 50, rand() % 125 + 50, 255);
            hostBlockTypes[i].albedo = make_uchar4(rand() % 55 + 200, rand() % 25 + 100, 0, 255); //make_uchar4(255, 12, 167, 255);
            hostBlockTypes[i].emittance = 35.0f * multiplier;
            hostBlockTypes[i].specularity = 0;
            hostBlockTypes[i].specularColor = make_uchar4(0, 0, 0, 0);
        }
    }

    for (int i = BLOCK_RES * BLOCK_RES * BLOCK_RES * 8; i < BLOCK_RES * BLOCK_RES * BLOCK_RES * 9; ++i)
    {
        hostBlockTypes[i].albedo = make_uchar4(0, 0, 0, 0);

        float x = (i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 8) % BLOCK_RES + 0.5;
        float y = ((i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 8) / BLOCK_RES) % BLOCK_RES + 0.5;
        float z = (i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 8) / (BLOCK_RES * BLOCK_RES) + 0.5;

        float dist = sqrtf((x - BLOCK_RES / 2) * (x - BLOCK_RES / 2) + (y - BLOCK_RES / 2) * (y - BLOCK_RES / 2) + (z - BLOCK_RES / 2) * (z - BLOCK_RES / 2));

        if (dist < 6)
        {
            //hostBlockTypes[i].albedo = make_uchar4(rand() % 200 + 55, rand() % 50, rand() % 125 + 50, 255);
            hostBlockTypes[i].albedo = make_uchar4(rand() % 5 + 15, rand() % 25 + 150, rand() % 55 + 200, 255); //make_uchar4(255, 12, 167, 255);
            hostBlockTypes[i].emittance = 35.0f * multiplier;
            hostBlockTypes[i].specularity = 0;
            hostBlockTypes[i].specularColor = make_uchar4(0, 0, 0, 0);
        }
    }

    for (int i = BLOCK_RES * BLOCK_RES * BLOCK_RES * 9; i < BLOCK_RES * BLOCK_RES * BLOCK_RES * 10; ++i)
    {
        hostBlockTypes[i].albedo = make_uchar4(0, 0, 0, 0);

        float x = (i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 9) % BLOCK_RES + 0.5;
        float y = ((i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 9) / BLOCK_RES) % BLOCK_RES + 0.5;
        float z = (i - BLOCK_RES * BLOCK_RES * BLOCK_RES * 9) / (BLOCK_RES * BLOCK_RES) + 0.5;

        float dist = sqrtf((x - BLOCK_RES / 2) * (x - BLOCK_RES / 2) + (y - BLOCK_RES / 2) * (y - BLOCK_RES / 2) + (z - BLOCK_RES / 2) * (z - BLOCK_RES / 2));

        if (dist < 6)
        {
            //hostBlockTypes[i].albedo = make_uchar4(rand() % 200 + 55, rand() % 50, rand() % 125 + 50, 255);
            hostBlockTypes[i].albedo = make_uchar4(rand() % 5 + 15, rand() % 55 + 200, 0, 255); //make_uchar4(255, 12, 167, 255);
            hostBlockTypes[i].emittance = 15.0f * multiplier;
            hostBlockTypes[i].specularity = 0;
            hostBlockTypes[i].specularColor = make_uchar4(0, 0, 0, 0);
        }
    }

    for (int i = BLOCK_RES * BLOCK_RES * BLOCK_RES * 10; i < BLOCK_RES * BLOCK_RES * BLOCK_RES * 11; ++i)
    {
        hostBlockTypes[i].albedo = make_uchar4(0, 0, 0, 0);
    }

    for (int x = 0; x < BLOCK_RES; ++x)
    {
        for (int z = 0; z < BLOCK_RES; ++z)
        {
            if (rand() % 5 > 2)
            {
                float rand0to1 = rand() % 10000 / 10000.0f;

                int height = rand0to1 * rand0to1 * BLOCK_RES;

                for (int y = 0; y < height; ++y)
                {
                    int voxelArrayPos = 10 * (BLOCK_RES * BLOCK_RES * BLOCK_RES) + z * (BLOCK_RES * BLOCK_RES) + y * BLOCK_RES + x;

                    hostBlockTypes[voxelArrayPos].albedo = make_uchar4(rand() % 20 + 15, rand() % 55 + 200, rand() % 30 + 20, 255); //make_uchar4(255, 12, 167, 255);
                    hostBlockTypes[voxelArrayPos].emittance = 0;
                    hostBlockTypes[voxelArrayPos].specularity = 0;
                    hostBlockTypes[voxelArrayPos].specularColor = make_uchar4(0, 0, 0, 0);
                }
            }
        }
    }

    for (int i = BLOCK_RES * BLOCK_RES * BLOCK_RES * 11; i < BLOCK_RES * BLOCK_RES * BLOCK_RES * 12; ++i)
    {
        if (rand() % 5 < 4)
        {
            int brightness = rand() % 75 + 50;
            hostBlockTypes[i].albedo = make_uchar4(brightness, brightness, brightness + 20, 255);
            hostBlockTypes[i].emittance = 0;
            hostBlockTypes[i].specularity = 0;
            hostBlockTypes[i].specularColor = make_uchar4(0, 0, 0, 0);
        }
        else if (rand() % 3 == 0)
        {
            int green = rand() % 50 + 150;
            hostBlockTypes[i].albedo = make_uchar4(green + 55, green, 0, 255);
            hostBlockTypes[i].emittance = 0;
            hostBlockTypes[i].specularity = (rand() % 1000) / 2000.0f + 0.3;
            hostBlockTypes[i].specularColor = make_uchar4(green + 55, green, 0, 255);
        }
    }

    for (int x = 19; x <= 41; ++x)
    {
        for (int y = 9; y <= 28; ++y)
        {
            if (rand() % 30 > 0 || y > 20)
            {
                PutBlock(hostBlocks, x, y, 41, 5);
            }
            else
            {
                PutBlock(hostBlocks, x, y, 41, 11);
            }
            
            if (rand() % 30 > 0 || y > 20)
            {
                PutBlock(hostBlocks, x, y, 19, 5);
            }
            else
            {
                PutBlock(hostBlocks, x, y, 19, 11);
            }
        }
    }

    for (int z = 19; z <= 41; ++z)
    {
        for (int y = 9; y <= 28; ++y)
        {
            if (rand() % 30 > 0 || y > 20)
            {
                PutBlock(hostBlocks, 19, y, z, 5);
            }
            else
            {
                PutBlock(hostBlocks, 19, y, z, 11);
            }

            if (rand() % 30 > 0 || y > 20)
            {
                PutBlock(hostBlocks, 41, y, z, 5);
            }
            else
            {
                PutBlock(hostBlocks, 41, y, z, 11);
            }
        }
    }

    for (int y = 10; y <= 25; ++y)
    {
        PutBlock(hostBlocks, 35, y, 36, 5);
        PutBlock(hostBlocks, 23, y, 24, 5);
        PutBlock(hostBlocks, 30, y, 30, 5);
    }

    int cachePosLUTBytes = BLOCK_TYPES * BLOCK_RES * BLOCK_RES * BLOCK_RES * VOXEL_FACES * sizeof(int);

    int* cachePosLUT = (int*)malloc(cachePosLUTBytes);

    for (int i = 1; i < 12; ++i)
    {
        cacheSizes[i] = 0;

        int count = 0;

        auto GetPosInArray = [](int i, int3 pos)
        {
            return (i * BLOCK_RES * BLOCK_RES * BLOCK_RES) + pos.z * BLOCK_RES * BLOCK_RES + pos.y * BLOCK_RES + pos.x;
        };

        auto GetPosInBlock = [](int3 pos)
        {
            return pos.z * BLOCK_RES * BLOCK_RES + pos.y * BLOCK_RES + pos.x;
        };

        auto CheckIfFull = [](int pos, Voxel* blockTypes)
        {
            return (blockTypes[pos].albedo.x != 0 || blockTypes[pos].albedo.y != 0 || blockTypes[pos].albedo.z != 0 || blockTypes[pos].albedo.w != 0);
        };

        auto CheckIfVoxelIsInsideBlock = [](int3 pos)
        {
            return (pos.x >= 0 && pos.x < BLOCK_RES && pos.y >= 0 && pos.y < BLOCK_RES && pos.z >= 0 && pos.z < BLOCK_RES);
        };

        auto GetNeighbour = [](int3 pos, int faceIndex)
        {
            int3 neighbour;

            switch (faceIndex)
            {
            case 0:
                neighbour = { pos.x - 1, pos.y, pos.z };
                break;
            case 1:
                neighbour = { pos.x + 1, pos.y, pos.z };
                break;
            case 2:
                neighbour = { pos.x, pos.y - 1, pos.z };
                break;
            case 3:
                neighbour = { pos.x, pos.y + 1, pos.z };
                break;
            case 4:
                neighbour = { pos.x, pos.y, pos.z - 1 };
                break;
            case 5:
                neighbour = { pos.x, pos.y, pos.z + 1 };
                break;
            }

            return neighbour;
        };

        bool visibleVoxels[BLOCK_RES * BLOCK_RES * BLOCK_RES];
        bool visibleFaces[BLOCK_RES * BLOCK_RES * BLOCK_RES * VOXEL_FACES];

        for (int z = 0; z < BLOCK_RES; ++z)
        {
            for (int y = 0; y < BLOCK_RES; ++y)
            {
                for (int x = 0; x < BLOCK_RES; ++x)
                {
                    visibleVoxels[GetPosInBlock({ x, y, z })] = false;

                    int pos = GetPosInArray(i, { x, y, z });

                    bool voxelIsFull = CheckIfFull(pos, hostBlockTypes);

                    for (int j = 0; j < VOXEL_FACES; ++j)
                    {
                        int faceInBlock = GetPosInBlock({ x, y, z }) * VOXEL_FACES + j;
                        int faceArrayIndex = pos * VOXEL_FACES + j;

                        visibleFaces[faceInBlock] = false;
                        cachePosLUT[faceArrayIndex] = -1;
                    }

                    if (voxelIsFull)
                    {
                        for (int j = 0; j < VOXEL_FACES; ++j)
                        {
                            int faceInBlock = GetPosInBlock({ x, y, z }) * VOXEL_FACES + j;
                            int faceArrayIndex = pos * VOXEL_FACES + j;

                            int3 neighbour = GetNeighbour({ x, y, z }, j);

                            if (CheckIfVoxelIsInsideBlock(neighbour))
                            {
                                if (CheckIfFull(GetPosInArray(i, neighbour), hostBlockTypes) == false)
                                {
                                    bool visitedVoxels[BLOCK_RES * BLOCK_RES * BLOCK_RES];

                                    for (int k = 0; k < BLOCK_RES * BLOCK_RES * BLOCK_RES; ++k)
                                    {
                                        visitedVoxels[k] = false;
                                    }

                                    if (LocateExitInBlock(i, neighbour, hostBlockTypes, visitedVoxels))
                                    {
                                        for (int k = 0; k < VOXEL_FACES; ++k)
                                        {
                                            if (visibleFaces[GetPosInBlock({ x, y, z }) * VOXEL_FACES + k] == false)
                                            {
                                                cachePosLUT[pos * VOXEL_FACES + k] = count;
                                            }
                                        }

                                        cachePosLUT[faceArrayIndex] = count;

                                        ++count;

                                        visibleVoxels[GetPosInBlock({ x, y, z })] = true;
                                        visibleFaces[faceInBlock] = true;
                                    }
                                }
                            }
                            else
                            {
                                for (int k = 0; k < VOXEL_FACES; ++k)
                                {
                                    if (visibleFaces[GetPosInBlock({ x, y, z }) * VOXEL_FACES + k] == false)
                                    {
                                        cachePosLUT[pos * VOXEL_FACES + k] = count;
                                    }
                                }

                                cachePosLUT[faceArrayIndex] = count;
                               
                                ++count;

                                visibleVoxels[GetPosInBlock({ x, y, z })] = true;
                                visibleFaces[faceInBlock] = true;
                            }
                        }
                    }
                    else
                    {
                        bool visitedVoxels[BLOCK_RES * BLOCK_RES * BLOCK_RES];

                        for (int k = 0; k < BLOCK_RES * BLOCK_RES * BLOCK_RES; ++k)
                        {
                            visitedVoxels[k] = false;
                        }

                        if (LocateExitInBlock(i, { x, y, z }, hostBlockTypes, visitedVoxels) == false)
                        {
                            hostBlockTypes[pos].albedo = make_uchar4(0, 0, 0, 255);
                        }
                    }
                }
            }
        }

        cacheSizes[i] = count;

        /*for (int z = 0; z < BLOCK_RES; ++z)
        {
            for (int y = 0; y < BLOCK_RES; ++y)
            {
                for (int x = 0; x < BLOCK_RES; ++x)
                {
                    if (visibleVoxels[GetPosInBlock({ x, y, z })] == false)
                    {
                        for (int j = 0; j < VOXEL_FACES; ++j)
                        {
                            int faceArrayIndex = GetPosInArray(i, { x, y, z }) * VOXEL_FACES + j;

                            int3 neighbour = GetNeighbour({ x, y, z }, j);

                            if (CheckIfVoxelIsInsideBlock(neighbour))
                            {
                                if (visibleVoxels[GetPosInBlock(neighbour)])
                                {
                                    for (int k = 0; k < VOXEL_FACES; ++k)
                                    {
                                        int index1 = GetPosInArray(i, { x, y, z }) * VOXEL_FACES + k;
                                        int index2 = GetPosInArray(i, neighbour) * VOXEL_FACES + k;

                                        cachePosLUT[index1] = cachePosLUT[index2];
                                    }

                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }*/
    }

    std::cout << "leaves cache size (bytes): " << cacheSizes[1] * sizeof(IrradianceCache) << '\n';
    std::cout << "dirt cache size (bytes): " << cacheSizes[2] * sizeof(IrradianceCache) << '\n';
    std::cout << "grass cache size (bytes): " << cacheSizes[3] * sizeof(IrradianceCache) << '\n';
    std::cout << "log cache size (bytes): " << cacheSizes[4] * sizeof(IrradianceCache) << '\n';
    std::cout << "stone cache size (bytes): " << cacheSizes[5] * sizeof(IrradianceCache) << '\n';
    std::cout << "lights cache size (bytes): " << cacheSizes[6] * sizeof(IrradianceCache) << '\n';

    int cacheOffset = 0;
    int blockCount = 0;

    for (int i = 0; i < WORLD_W * WORLD_L * WORLD_H; ++i)
    {
        int blockId = hostBlocks[i].id;

        if (blockId != 0)
        {
            hostBlocks[i].irradianceCacheOffset = cacheOffset;

            cacheOffset += cacheSizes[blockId];
            ++blockCount;
        }
    }

    cachedFaces = cacheOffset;

    int irradianceCacheBytes = MAX_IRRADIANCE_FACES * sizeof(IrradianceCache);

    std::cout << "number of cached faces: " << cachedFaces << '\n';
    std::cout << "size of irradiance cache (bytes): " << irradianceCacheBytes << '\n';
    std::cout << "number of blocks: " << blockCount << '\n';

    IrradianceCache* hostIrradianceCache = (IrradianceCache*)malloc(irradianceCacheBytes);

    for (int i = 0; i < MAX_IRRADIANCE_FACES; ++i)
    {
        hostIrradianceCache[i].irradiance = ZERO_VEC3D;
        hostIrradianceCache[i].bounceCount = 1;
        hostIrradianceCache[i].currentUpdate = 0;
    }

    Light* deviceLights;
    cudaMalloc(&deviceLights, MAX_LIGHTS_PER_LIGHT_CHUNK * LIGHT_CHUNKS_W * LIGHT_CHUNKS_H * LIGHT_CHUNKS_L);

    IrradianceCache* deviceIrradianceCache;
    cudaMalloc(&deviceIrradianceCache, irradianceCacheBytes);
    cudaMemcpy(deviceIrradianceCache, hostIrradianceCache, irradianceCacheBytes, cudaMemcpyHostToDevice);

    free(hostIrradianceCache);

    Block* deviceBlocks;
    cudaMalloc(&deviceBlocks, blockBytes);
    cudaMemcpy(deviceBlocks, hostBlocks, blockBytes, cudaMemcpyHostToDevice);

    //free(hostBlocks);

    Voxel* deviceBlockTypes;
    cudaMalloc(&deviceBlockTypes, blockTypeBytes);
    cudaMemcpy(deviceBlockTypes, hostBlockTypes, blockTypeBytes, cudaMemcpyHostToDevice);

    int chunkCount = (WORLD_W * WORLD_H * WORLD_L) / (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE);

    for (int i = 0; i < chunkCount; ++i)
    {
        UpdateChunkDistanceField(i, deviceBlocks);
    }

    free(hostBlockTypes);

    int* deviceCachePosLUT;
    cudaMalloc(&deviceCachePosLUT, cachePosLUTBytes);
    cudaMemcpy(deviceCachePosLUT, cachePosLUT, cachePosLUTBytes, cudaMemcpyHostToDevice);

    free(cachePosLUT);

    int* lightsPerLightChunk;
    cudaMalloc(&lightsPerLightChunk, LIGHT_CHUNKS_W * LIGHT_CHUNKS_H * LIGHT_CHUNKS_L * sizeof(int));
    cudaMemset(lightsPerLightChunk, 0, LIGHT_CHUNKS_W * LIGHT_CHUNKS_H * LIGHT_CHUNKS_L * sizeof(int));

    WorldData worldData = { deviceBlocks, deviceBlockTypes, deviceIrradianceCache, deviceCachePosLUT, deviceLights, lightsPerLightChunk };

    AddLightSource({ { 30.5f, 27.5f, 30.5f }, 0.35, 1.0 }, worldData);
    AddLightSource({ { 35.5f, 27.5f, 39.5f }, 0.35, 1.0 }, worldData);
    AddLightSource({ { 39.5f, 27.5f, 23.5f }, 0.35, 1.0 }, worldData);
    AddLightSource({ { 20.5f, 10.5f, 40.5f }, 0.35, 1.0 }, worldData);

    ReflectionPoint* hostReflectionPoints = (ReflectionPoint*)malloc(sizeof(ReflectionPoint) * FORWARD_PATH_COUNT);

    for (int i = 0; i < FORWARD_PATH_COUNT; ++i)
    {
        hostReflectionPoints[i].irradianceCachePos = -1;
        hostReflectionPoints[i].continuationProbability = 0;
    }

    ReflectionPoint* deviceReflectionPoints;
    cudaMalloc(&deviceReflectionPoints, sizeof(ReflectionPoint) * FORWARD_PATH_COUNT);
    cudaMemcpy(deviceReflectionPoints, hostReflectionPoints, sizeof(ReflectionPoint) * FORWARD_PATH_COUNT, cudaMemcpyHostToDevice);


    std::mt19937 randomEngine(seedEngine());

    RandState* hostRandStates = (RandState*)malloc(sizeof(RandState) * FORWARD_PATH_COUNT);

    for (int i = 0; i < FORWARD_PATH_COUNT; ++i)
    {
        hostRandStates[i].m_w = randomEngine() % 65536;
        hostRandStates[i].m_z = randomEngine() % 65536;
    }

    RandState* deviceRandStates;
    cudaMalloc(&deviceRandStates, sizeof(RandState) * FORWARD_PATH_COUNT);
    cudaMemcpy(deviceRandStates, hostRandStates, sizeof(RandState) * FORWARD_PATH_COUNT, cudaMemcpyHostToDevice);

    /* Loop until the user closes the window !glfwWindowShouldClose(window) */
    while (!glfwWindowShouldClose(window))
    {
        double initialTime = glfwGetTime();
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        Input(window);
        Simulate(hostBlocks, worldData);
        Render(gBuffer, worldData, deviceReflectionPoints, deviceRandStates);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();

        double finalTime = glfwGetTime();

        elapsedTime = finalTime - initialTime;

        timeCounter += elapsedTime;
        frameCounter += 1;

        if (timeCounter > 1.0)
        {
            std::cout << "FPS: " << frameCounter << '\n';
            std::cout << "ms/frame: " << 1000.0 / float(frameCounter) << '\n';

            timeCounter = 0;
            frameCounter = 0;
        }
    }

    glfwTerminate();

    return 0;
}