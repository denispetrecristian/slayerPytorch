/*
Author: Denis-Cristian Petre

This header contains methods to encode Image Tensors into Spike Tensors
*/
#ifndef ENCODINGKERNELS_H_INCLUDED
#define ENCODINGKERNELS_H_INCLUDED

#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

template <class T>
__global__ void rateEncodingKernel(T *__restrict__ output, const T *__restrict__ input, unsigned nNeurons, int tSample)
{

    unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuronID > nNeurons)
        return;

    unsigned value = input[neuronID];
    unsigned interval = floor(tSample / value);

    for (unsigned i = 0; i < tSample; i++)
    {
        if (i % interval == 0)
        {
            output[neuronID * tSample + i] = 1;
        }
    }
}

template <class T>
__global__ void poissonEncodingKernel(T *__restrict__ output, const T *__restrict__ input, unsigned nNeurons, int tSample)
{
    unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;
    float value = input[neuronID];
    curandState state;
    curand_init(value * 1000, neuronID, 0, &state);

    if (neuronID > nNeurons)
        return;
    for (unsigned i = 0; i < tSample; i++)
    {
        float unifRand = curand_uniform(&state);
        if (unifRand < value)
        {
            output[neuronID * tSample + i] = 1;
        }
    }
}

template <class T>
void poisson(T *output, const T *input, unsigned nNeurons, int tSample)
{
    // This code assumes the Ts to be equal to 1
    unsigned thread = 256;
    unsigned block = ceil(1.0f * nNeurons / thread);

    poissonEncodingKernel<T><<<block, thread>>>(output, input, nNeurons, tSample);
}

template <class T>
void rate(T *output, const T *input, unsigned nNeurons, int tSample)
{
    // This code assumes the Ts to be equal to 1
    unsigned thread = 256;
    unsigned block = ceil(1.0f * nNeurons / thread);
    rateEncodingKernel<T><<<block, thread>>>(output, input, nNeurons, tSample);
}

#endif
