#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define TYPE float

struct layer
{
    char *name;
    struct layer* next;

    //layer
    int ndim;
    int *input_shape;
    int *output_shape;

    // pooling param
    int ksize;
    int pad;
    int stride;

} layer_t;

layer_t *init_pooling(const int ndim, const int *input_shape,
                      const int ksize, const int pad, const int stride)
{
    assert(ndim>0 && input_shape && ksize>0 && pad>=0 && stride>0);
    layer_t *l = calloc(1, sizeof(layer_t));
    
    return l;
}

layer_t *forward_pooling(layer_t *l)
{

}

int main(int argc, char *argv[])
{
    
    return 0;
}
