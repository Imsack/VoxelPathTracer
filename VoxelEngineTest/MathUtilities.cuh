#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <math.h>

#define TAU 6.2831853072
#define PI 3.1415926536

#define ZERO_VEC3D { 0, 0, 0 }

__host__ __device__ float Sqrt(float x)
{
	return sqrtf(x);
}

__host__ __device__ float Cbrt(float x)
{
	return cbrtf(x);
}

__host__ __device__ float Min(float x, float y)
{
	return fminf(x, y);
}

__host__ __device__ float Max(float x, float y)
{
	return fmaxf(x, y);
}

__host__ __device__ float Abs(float x)
{
	return fabsf(x);
}

__host__ __device__ float Clamp(float x, float lower, float upper)
{
	return Max(lower, Min(upper, x));
}

__host__ __device__ float AbsMin(float x, float y)
{
	return (Abs(x) < Abs(y) ? x : y);
}

__host__ __device__ float Ceil(float x)
{
	return ceilf(x);
}

__host__ __device__ float Floor(float x)
{
	return floorf(x);
}

__device__ float FastSqrt(float x)
{
	return __fsqrt_rd(x);
}

__device__ float FastSin(float x)
{
	return __sinf(x);
}

__device__ float FastCos(float x)
{
	return __cosf(x);
}

__device__ void FastSinCos(float x, float* sin, float* cos)
{
	__sincosf(x, sin, cos);
}

__device__ float FastReciprocal(float x)
{
	return __frcp_ru(x);
}

__device__ float FastDivide(float x, float y)
{
	return __fdividef(x, y);
}

__device__ float FastReciprocalSqrt(float x)
{
	return __frsqrt_rn(x);
}

__device__ float ACEScurve(float x)
{
	//return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
	return tanh(x);
}

__device__ float GammaCorrection(float x)
{
	float output = 12.92 * x;

	if (x > 0.0031308)
	{
		output = 1.055 * __powf(x, 1.0 / 2.4) - 0.055;
	}

	return output;
}

struct Vec2D
{
	float x, y;
};

__host__ __device__ Vec2D R2SequenceSample(int index)
{
	// https://observablehq.com/@jrus/plastic-sequence source

	double p1 = 0.7548776662466927;
	double p2 = 0.5698402909980532;

	double a = p1 * index;
	double b = p2 * index;

	return { float(a - int(a)), float(b - int(b)) };
}

struct Vec3D
{
	float x, y, z;
};

__host__ __device__ Vec3D operator + (Vec3D v1, Vec3D v2)
{
	return { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z };
}

__host__ __device__ Vec3D operator - (Vec3D v1, Vec3D v2)
{
	return { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
}

__host__ __device__ Vec3D operator * (Vec3D v, float s)
{
	return { v.x * s, v.y * s, v.z * s };
}

__host__ __device__ Vec3D operator * (float s, Vec3D v)
{
	return { v.x * s, v.y * s, v.z * s };
}

__host__ __device__ Vec3D operator * (Vec3D v1, Vec3D v2)
{
	return { v1.x * v2.x, v1.y * v2.y, v1.z * v2.z };
}

__host__ __device__ void operator += (Vec3D* v1, Vec3D v2)
{
	v1->x += v2.x;
	v1->y += v2.y;
	v1->z += v2.z;
}

__host__ __device__ void operator -= (Vec3D* v1, Vec3D v2)
{
	v1->x -= v2.x;
	v1->y -= v2.y;
	v1->z -= v2.z;
}

__host__ __device__ void operator *= (Vec3D* v1, Vec3D v2)
{
	v1->x *= v2.x;
	v1->y *= v2.y;
	v1->z *= v2.z;
}

__host__ __device__ bool operator == (Vec3D v1, Vec3D v2)
{
	return v1.x == v2.x && v1.y == v2.y && v1.z == v2.z;
}

__host__ __device__ uint4 operator + (uint4 a, uint4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ uint4 operator + (uchar4 a, uchar4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ uint4 operator + (uint4 a, uchar4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ uint4 operator + (uchar4 a, uint4 b)
{
	return make_uint4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__host__ __device__ void Scale(Vec3D* v, float s)
{
	v->x *= s;
	v->y *= s;
	v->z *= s;
}

__host__ __device__ float Dot(Vec3D v1, Vec3D v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ Vec3D Cross(Vec3D v1, Vec3D v2)
{
	return { v1.y * v1.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x };
}

__host__ __device__ float Magnitude(Vec3D v)
{
	return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float FastMagnitude(Vec3D v)
{
	return __fsqrt_rd(v.x * v.x + v.y * v.y + v.z * v.z);
}

__host__ __device__ float MagnitudeSquared(Vec3D v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}

__host__ __device__ Vec3D Normalize(Vec3D v)
{
	float reciprocalMagnitude = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

	return { v.x * reciprocalMagnitude, v.y * reciprocalMagnitude, v.z * reciprocalMagnitude };
}

__host__ __device__ void Normalize(Vec3D* v)
{
	float reciprocalMagnitude = 1.0f / sqrtf(v->x * v->x + v->y * v->y + v->z * v->z);

	v->x *= reciprocalMagnitude;
	v->y *= reciprocalMagnitude;
	v->z *= reciprocalMagnitude;
}

__device__ Vec3D FastNormalize(Vec3D v)
{
	float reciprocalMagnitude = __frsqrt_rn(v.x * v.x + v.y * v.y + v.z * v.z);

	return { v.x * reciprocalMagnitude, v.y * reciprocalMagnitude, v.z * reciprocalMagnitude };
}

__device__ void FastNormalize(Vec3D* v)
{
	float reciprocalMagnitude = __frsqrt_rn(v->x * v->x + v->y * v->y + v->z * v->z);

	v->x *= reciprocalMagnitude;
	v->y *= reciprocalMagnitude;
	v->z *= reciprocalMagnitude;
}

__host__ __device__ Vec3D Lerp(Vec3D v1, Vec3D v2, float t)
{
	return v1 + t * (v2 - v1);
}

__device__ float Proj(Vec3D v1, Vec3D v2)
{
	return FastDivide(Dot(v1, v2), MagnitudeSquared(v2));
}

__host__ __device__ float ManhattanDistance(Vec3D v1, Vec3D v2)
{
	return Max(Abs(v1.x - v2.x), Max(Abs(v1.y - v2.y), Abs(v1.z - v2.z)));
}

__device__ Vec3D ToneMapper(Vec3D color)
{
	float shrinkFactor = 0;

	float maxChannel = Max(color.x, Max(color.y, color.z));

	if (maxChannel > 0)
	{
		shrinkFactor = FastDivide((1 + 0.1 * __logf(10 * maxChannel + 1) - __expf(-maxChannel)), maxChannel);
		//shrinkFactor = FastDivide(ACEScurve(maxChannel), maxChannel);
	}

	return color * shrinkFactor;
}

struct Quaternion
{
	double realPart;
	Vec3D vecPart;
};

__host__ __device__ Quaternion CreateRotQuat(Vec3D axis, float angle)
{
	return { cosf(angle * 0.5), axis * sinf(angle * 0.5) };
}

__host__ __device__ Quaternion Conjugate(Quaternion q)
{
	return { q.realPart, { -q.vecPart.x, -q.vecPart.y, -q.vecPart.z } };
}

__host__ __device__ Quaternion operator * (Quaternion q1, Quaternion q2)
{
	Quaternion result = { 0, ZERO_VEC3D };

	result.vecPart.x = q1.vecPart.x * q2.realPart + q1.vecPart.y * q2.vecPart.z - q1.vecPart.z * q2.vecPart.y + q1.realPart * q2.vecPart.x;
	result.vecPart.y = -q1.vecPart.x * q2.vecPart.z + q1.vecPart.y * q2.realPart + q1.vecPart.z * q2.vecPart.x + q1.realPart * q2.vecPart.y;
	result.vecPart.z = q1.vecPart.x * q2.vecPart.y - q1.vecPart.y * q2.vecPart.x + q1.vecPart.z * q2.realPart + q1.realPart * q2.vecPart.z;
	result.realPart = -q1.vecPart.x * q2.vecPart.x - q1.vecPart.y * q2.vecPart.y - q1.vecPart.z * q2.vecPart.z + q1.realPart * q2.realPart;

	return result;
}

__host__ __device__ void Normalize(Quaternion* q)
{
	double reciprocalLength = 1.0f / sqrtf(q->realPart * q->realPart + q->vecPart.x * q->vecPart.x + q->vecPart.y * q->vecPart.y + q->vecPart.z * q->vecPart.z);

	q->realPart *= reciprocalLength;
	q->vecPart.x *= reciprocalLength;
	q->vecPart.y *= reciprocalLength;
	q->vecPart.z *= reciprocalLength;
}

struct Matrix3X3
{
	Vec3D iHat;
	Vec3D jHat;
	Vec3D kHat; 
};

__host__ __device__ Vec3D operator * (Matrix3X3 m, Vec3D v)
{
	return v.x * m.iHat + v.y * m.jHat + v.z * m.kHat;
}

struct RandState
{
	uint32_t m_z;
	uint32_t m_w;
};

__host__ __device__ int RandomUnsignedInt32(RandState* randState)
{
	randState->m_z = 36969 * (randState->m_z & 65535) + (randState->m_z >> 16);
	randState->m_w = 18000 * (randState->m_w & 65535) + (randState->m_w >> 16);

	return ((randState->m_z << 16) + randState->m_w);
}

__host__ __device__ float RandomFloat0To1(RandState* randState)
{
	randState->m_z = 36969 * (randState->m_z & 65535) + (randState->m_z >> 16);
	randState->m_w = 18000 * (randState->m_w & 65535) + (randState->m_w >> 16);

	return ((randState->m_z << 16) + randState->m_w) * 2.328306435454494e-10;
}