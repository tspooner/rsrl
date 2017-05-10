#ifndef _TILES_H_
#define _TILES_H_

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#define MAX_NUM_VARS 20        // Maximum number of variables in a grid-tiling
#define MAX_NUM_COORDS 100     // Maximum number of hashing coordinates
#define MaxLONGINT 2147483647

void tiles(
    size_t the_tiles[],           // provided array contains returned tiles (tile indices)
    int num_tilings,           // number of tile indices to be returned in tiles
    int memory_size,           // total number of possible tiles
    double floats[],           // array of floating point variables
    int num_floats,            // number of floating point variables
    int ints[],                // array of integer variables
    int num_ints);             // number of integer variables

int hash_UNH(int *ints, int num_ints, long m, int increment);

#endif
