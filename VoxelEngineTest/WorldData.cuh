#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "MathUtilities.cuh"

#define SCREEN_W 1920 // 1280
#define SCREEN_H 1080 // 960

#define WORLD_W 64
#define WORLD_H 64
#define WORLD_L 64
#define CHUNK_SIZE 16
#define LIGHT_CHUNK_SIZE 8

#define LIGHT_CHUNKS_W (WORLD_W / LIGHT_CHUNK_SIZE)
#define LIGHT_CHUNKS_H (WORLD_H / LIGHT_CHUNK_SIZE)
#define LIGHT_CHUNKS_L (WORLD_L / LIGHT_CHUNK_SIZE)

#define BLOCK_TYPES 32
#define BLOCK_RES 16
#define VOXEL_FACES 6

//#define MAX_SAMPLES 128
#define MAX_IRRADIANCE_FACES (768 * 256 * 256) // 100 663 296
#define MAX_IRRADIANCE_SAMPLES 131072
#define DEAMPLIFIED_BOUNCE_COUNT 1

#define MAX_LIGHTS_PER_LIGHT_CHUNK 256
#define PROPOSAL_LIGHT_SAMPLES 16

#define FORWARD_PATH_COUNT 65536

float elapsedTime;
float timeCounter;
int frameCounter;

Matrix3X3* orientationMatrices;
Vec3D* playerPositions;

enum InputFlag
{
	IDLE,
	PLACE_BLOCK,
	REMOVE_BLOCK
};

bool Q_REPEAT = false;
bool E_REPEAT = false;

int inventoryId = 6;

struct Player
{
private:
	float movementSpeed = 3.0;
	float rotationSpeed = 4.0;
	float zoomSpeed = 1.0;

public:
	Vec3D pos;
	Quaternion orientation;
	float FOV;
	InputFlag inputFlag;

	Player(Vec3D pos, Quaternion orientation, float FOV)
	{
		this->pos = pos;
		this->orientation = orientation;
		this->FOV = FOV;
	}

	float MovementSpeed(float elapsedTime)
	{
		return movementSpeed * elapsedTime;
	}

	float RotationSpeed(float elapsedTime)
	{
		return rotationSpeed * elapsedTime;
	}

	float ZoomSpeed(float elapsedTime)
	{
		return zoomSpeed * elapsedTime;
	}
};

struct Chunk
{
	int index;
};

struct Block // made from BLOCK_RES * BLOCK_RES * BLOCK_RES number of voxels
{
	uint16_t id;
	uint8_t distance;
	int irradianceCacheOffset;
};

struct Voxel
{
	float emittance = 0;
	uchar4 albedo;
	uchar4 specularColor = make_uchar4(0, 0, 0, 0);
	float specularity = 0;
};

struct IrradianceCache
{
	Vec3D irradiance;
	int bounceCount;
	int currentUpdate;
};

struct Light
{
	Vec3D position;
	float radius;
	float emittance;
};

struct ReflectionPoint
{
	Vec3D pos;
	Matrix3X3 tangentMatrix;
	Vec3D albedo;
	int irradianceCachePos;
	float continuationProbability;
};

struct WorldData
{
	Block* blocks;
	Voxel* blockTypes;
	IrradianceCache* irradianceCache;
	int* cachePosLUT;
	Light* lights;
	int* lightsPerLightChunk;
};

struct GBuffer
{
	Vec3D* frameBuffer;
	Vec3D* frameBufferCopy;
	Vec3D* albedoBuffer;
	//RandState* randStates;
	int* primaryIrradianceCachePositions;
	//IrradianceCache* irradianceBuffer;
	bool* updateVoxel;
	Vec3D* specularBuffer;
	float* specularityBuffer;
};