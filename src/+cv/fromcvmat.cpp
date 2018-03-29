/**
 * @file fromcvmat.cpp
 * @brief mex interface for load opencv binary mat file
 * @ingroup core
 */
#include "mexopencv.hpp"

#include <vector>
#include <string.h>

using namespace std;
using namespace cv;

// this version is slow as hell
static mxArray* slowLoadMatBinary(const mxArray* mat)
{
    Mat in_mat;
    const Mat buf = MxArray(mat).toMat();
    if (buf.isContinuous() == false || buf.type() != CV_8U) {
        mexErrMsgIdAndTxt("mexopencv:error", "fromcvmat: bad data");
        return NULL;
    }
    if (buf.total() < sizeof(int)*3) {
        mexErrMsgIdAndTxt("mexopencv:error", "fromcvmat: bad data");
        return NULL;
    }
    const char* p = (const char*)buf.data;
    int rows = *(int*)p; p += sizeof(int);
    if (rows == 0) return MxArray(in_mat);
    int cols = *(int*)p; p += sizeof(int);
    int type = *(int*)p; p += sizeof(int);
    in_mat.create(rows, cols, type);
    int bytes = (int)(in_mat.total()*in_mat.elemSize());
    if (buf.total() != bytes + sizeof(int)*3) {
        mexErrMsgIdAndTxt("mexopencv:error", "fromcvmat: bad data");
        return NULL;
    }
    memcpy(in_mat.data, p, bytes);
    return MxArray(in_mat);
}

static const ConstMap<int, mxClassID> ClassOf = ConstMap<int, mxClassID>
    (CV_64F, mxDOUBLE_CLASS)
    (CV_32F, mxSINGLE_CLASS)
    (CV_8S,  mxINT8_CLASS)
    (CV_8U,  mxUINT8_CLASS)
    (CV_16S, mxINT16_CLASS)
    (CV_16U, mxUINT16_CLASS)
    (CV_32S, mxINT32_CLASS);

static mxArray* loadMatBinary(const mxArray* m)
{
    if (!mxIsNumeric(m) || mxIsSparse(m) || mxIsComplex(m)) {
        mexErrMsgIdAndTxt("mexopencv:error", "fromcvmat: bad data");
        return NULL;
    }
    if (mxGetClassID(m) != mxUINT8_CLASS) {
        mexErrMsgIdAndTxt("mexopencv:error", "fromcvmat: bad data");
        return NULL;
    }
    mwSize ndims = mxGetNumberOfDimensions(m);
    if (ndims != 2) {
        mexErrMsgIdAndTxt("mexopencv:error", "fromcvmat: bad data");
        return NULL;
    }
    int bytes = (int)mxGetDimensions(m)[0];
    if (bytes < sizeof(int)*3) {
        mexErrMsgIdAndTxt("mexopencv:error", "fromcvmat: bad data");
        return NULL;
    }
    const char* mp = (const char*)mxGetData(m);
    int rows = *(int*)mp; mp += sizeof(int);
    if (rows == 0) return NULL;
    int cols = *(int*)mp; mp += sizeof(int);
    int type = *(int*)mp; mp += sizeof(int);
    int channels = CV_MAT_CN(type);
    int elemSize = CV_ELEM_SIZE(type);
    if (bytes != sizeof(int)*3 + rows*cols*elemSize) {
        mexErrMsgIdAndTxt("mexopencv:error", "fromcvmat: bad data");
        return NULL;
    }
    int depth = CV_MAT_DEPTH(type);
    int componentSz;
    switch (depth) {
    case CV_8U:
    case CV_8S:
        componentSz = 1;
        break;
    case CV_32S:
    case CV_32F:
        componentSz = 4;
        break;
    default:
        return slowLoadMatBinary(m);
    }
    mxClassID classId = ClassOf[depth];
    mwSize rdims[] = { rows, cols, channels };
    mxArray* r = mxCreateNumericArray(3, rdims, classId, mxREAL);
    char* p = (char*)mxGetData(r);
    switch (componentSz) {
    case 1: {
        uint8_t* pp = (uint8_t*)p;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                for (int c = 0; c < channels; ++c) {
                    *(pp + y + x*rows + c*rows*cols) = *(uint8_t*)mp;
                    mp += 1;
                }
            }
        }
        break;
    }
    case 4: {
        uint32_t* pp = (uint32_t*)p;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                for (int c = 0; c < channels; ++c) {
                    *(pp + y + x*rows + c*rows*cols) = *(uint32_t*)mp; 
                    mp += 4;
                }
            }
        }
        break;
    }
    default:
        mxDestroyArray(r);
        slowLoadMatBinary(m);
        return NULL;
    } 
    return r;
}

/**
 * Main entry called from Matlab
 * @param nlhs number of left-hand-side arguments
 * @param plhs pointers to mxArrays in the left-hand-side
 * @param nrhs number of right-hand-side arguments
 * @param prhs pointers to mxArrays in the right-hand-side
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check the number of arguments
    nargchk(nrhs==1 && nlhs<=1);

    // Process
    plhs[0] = loadMatBinary(prhs[0]);
}
