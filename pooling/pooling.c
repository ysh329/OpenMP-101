#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <float.h>

#define DTYPE float

// define structures
typedef struct layer
{
    char *name;
    struct layer* next;

    // basic
    DTYPE *input;
    DTYPE *output;
    int input_count;
    int ndim;
    int *input_shape;
    int *output_shape;
    int output_count;

    // pooling param
    int ksize;
    int pad;
    int stride;

} layer_t;

// define functions
layer_t *init_pooling(const char *name,
                      const int ndim, const int *input_shape,
                      const int ksize, const int pad, const int stride);
void print_pooling(layer_t *l);
layer_t *forward_pooling(layer_t *l);
void print_pooling(layer_t *l);
void destroy_pooling(layer_t *l);


layer_t *init_pooling(const char *name,
                      const int ndim, const int *input_shape,
                      const int ksize, const int pad, const int stride)
{
    assert(ndim>0 && input_shape && ksize>0 && pad>=0 && stride>0);

    layer_t *l         = calloc(1, sizeof(layer_t));
    l->name = calloc(strlen(name), sizeof(char));
    strcpy(l->name, name);
    l->next            = NULL;
    int input_count    = 1;
    for(int idx=0; idx<ndim; ++idx)
    {
        input_count *= input_shape[idx];
    }
    l->input           = calloc(input_count, sizeof(DTYPE));
    l->input_count     = input_count;
    l->ndim            = ndim;
    l->input_shape     = calloc(ndim, ndim*sizeof(int));
    l->output_shape    = calloc(ndim, ndim*sizeof(int));
    memcpy(l->input_shape, input_shape, ndim*sizeof(int));
    memcpy(l->output_shape, input_shape, ndim*sizeof(int));

    l->output_shape[2] = (l->input_shape[2] + 2*pad - ksize) / stride + 1;
    l->output_shape[3] = (l->input_shape[3] + 2*pad - ksize) / stride + 1;
    l->output_count    = 1;
    for(int idx=0; idx<ndim; ++idx)
    {
        l->output_count *= l->output_shape[idx];
    }
    l->output          = calloc(l->output_count, sizeof(DTYPE));

    l->ksize           = ksize;
    l->pad             = pad;
    l->stride          = stride;
    
    return l;
}

void print_pooling(layer_t *l)
{
    assert(l);
    printf("---- print_pooling ----\n");
    printf("l->name:%s\n", l->name);
    printf("l->input_shape:"); 
    for(int idx=0; idx<l->ndim; ++idx) 
        printf("%d ", l->input_shape[idx]);
    printf("\n");
    printf("l->output_shape:"); 
    for(int idx=0; idx<l->ndim; ++idx) 
        printf("%d ", l->output_shape[idx]);
    printf("\n");
    printf("l->input_count:%d\n",  l->input_count);
    printf("l->output_count:%d\n",  l->output_count);
    printf("l->ksize:%d\n",  l->ksize);
    printf("l->pad:%d\n",    l->pad);
    printf("l->stride:%d\n", l->stride);
    printf("\n");
    return;
}

// refer darknet implementation
#define OUT_IDX(n, c, h, w) l->output[n*l->output_shape[1]*l->output_shape[2]*l->output_shape[3]+\
                                      c*l->output_shape[2]*l->output_shape[3]+\
                                      h*l->output_shape[3]+\
                                      w]
#define IN_IDX(n, c, h, w)  l->input[n*l->input_shape[1]*l->input_shape[2]*l->input_shape[3]+\
                                     c*l->input_shape[2]*l->input_shape[3]+\
                                     h*l->input_shape[3]+\
                                     w]
layer_t *forward_pooling(layer_t *l)
{
    assert(l);
    int h_offset = -l->pad;
    int w_offset = -l->pad;
    #pragma omp parallel for
    for(int b=0; b<l->output_shape[0]; ++b)
    {
        for(int k=0; k<l->output_shape[1]; ++k)
        {
            for(int i=0; i<l->output_shape[2]; ++i)
            {
                for(int j=0; j<l->output_shape[3]; ++j)
                {
                    int out_idx = OUT_IDX(b, k, i, j);
                    int max_idx = -1;
                    DTYPE max = -FLT_MAX;
                    for(int n=0; n<l->ksize; ++n)
                    {
                        for(int m=0; m<l->ksize; ++m)
                        {
                            int cur_h  = i*l->stride + h_offset + n;
                            int cur_w  = j*l->stride + w_offset + m;
                            int in_idx = IN_IDX(b, k, cur_h, cur_w);

                            int valid  = (cur_h>=0 && cur_h<l->output[2] &&
                                          cur_w>=0 && cur_w<l->output[3]);
                            DTYPE val  = (valid!=0) ? l->input[in_idx] : -FLT_MAX;

                            max        = (val > max) ? val : max;
                            max_idx    = (val > max) ? in_idx: max_idx;
                        }
                    }
                    l->output[out_idx] = max;
                    // l->indexes[out_idx] = max_idx; // record for backprop
                }
            }
        }
    }
    return l;
}

void destroy_pooling(layer_t *l)
{
    assert(l &&
           l->input &&
           l->output &&
           l->input_shape &&
           l->output_shape);
    if(l->input)        free(l->input);        l->input        = NULL;
    if(l->input_shape)  free(l->input_shape);  l->input_shape  = NULL;
    if(l->output)       free(l->output);       l->output       = NULL;
    if(l->output_shape) free(l->output_shape); l->output_shape = NULL;
    if(l->next)         free(l->next);         l->next         = NULL; // TODO: loop free layer-wise
    if(l)               free(l);               l               = NULL;
    return;
}

int main(int argc, char *argv[])
{
    // init
    const char *name = "pooling_layer";
    const int ndim   = 4;
    const int ksize  = 2;
    const int pad    = 1;
    const int stride = 2;
    int *input_shape = calloc(ndim, sizeof(int));
    input_shape[0]   = 1;
    input_shape[1]   = 3;
    input_shape[2]   = 224;
    input_shape[3]   = 224;

    layer_t *l = init_pooling(name, ndim, input_shape, ksize, pad, stride);
    print_pooling(l);

    // forward pooling
    l = forward_pooling(l);

    // free
    destroy_pooling(l);
    if(input_shape) free(input_shape); input_shape = NULL;

    return 0;
}
