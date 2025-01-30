//
// Created by damitha on 1/29/25.
//

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <omp.h>

void fillRandom(float* array, int totalDim){
#pragma omp parallel for
	for (int i = 0; i < totalDim; i++) {
		array[i] = rand() % 10;
	}
}

#endif //UTILS_H
