/**
 * @file fromcvmat.cpp
 * @brief mex interface for save opencv binary mat file
 * @ingroup core
 */
#include "mexopencv.hpp"

#include <vector>
#include <string.h>

using namespace std;
using namespace cv;

// this version is slow as hell
static mxArray* slowSaveMatBinary(const mxArray* m)
{
    Mat buf;
    const Mat out_mat = MxArray(m).toMat();
    if (out_mat.empty()) {
        buf.create(sizeof(int)*3, 1, CV_8U);
        *(int*)buf.data = 0;
        return MxArray(buf);
    }
    int bytes = (int)(out_mat.total()*out_mat.elemSize());
    buf.create(sizeof(int)*3 + bytes, 1, CV_8U);
    char* p = (char*)buf.data;
    *(int*)p = out_mat.rows; p += sizeof(int);
    *(int*)p = out_mat.cols; p += sizeof(int);
    *(int*)p = out_mat.type(); p += sizeof(int);
    memcpy(p, out_mat.data, bytes);
    return MxArray(buf);
}

static const ConstMap<mxClassID, int> DepthOf = ConstMap<mxClassID, int>
    (mxDOUBLE_CLASS,  CV_64F)
    (mxSINGLE_CLASS,  CV_32F)
    (mxINT8_CLASS,    CV_8S)
    (mxUINT8_CLASS,   CV_8U)
    (mxINT16_CLASS,   CV_16S)
    (mxUINT16_CLASS,  CV_16U)
    (mxINT32_CLASS,   CV_32S)
    (mxUINT32_CLASS,  CV_32S)
    (mxLOGICAL_CLASS, CV_8U);

static mxArray* saveMatBinary(const mxArray* m)
{
    if (!mxIsNumeric(m) || mxIsSparse(m) || mxIsComplex(m)) {
        mexErrMsgIdAndTxt("mexopencv:error", "tocvmat: bad data");
    }
    mwSize ndims = mxGetNumberOfDimensions(m);
    if (ndims != 2 && ndims != 3) {
        return slowSaveMatBinary(m);
    }
    int rows, cols, channels;
    const mwSize* dims = mxGetDimensions(m);
    if (ndims == 2) {
        rows = (int)dims[0];
        cols = (int)dims[1];
        channels = 1;
    } else {
        rows = (int)dims[0];
        cols = (int)dims[1];
        channels = (int)dims[2];
    }
    mxClassID classId = mxGetClassID(m);
    int componentSz;
    switch (classId) {
    case mxUINT8_CLASS:
    case mxINT8_CLASS:
        componentSz = 1;
        break;
    case mxUINT32_CLASS:
    case mxINT32_CLASS:
    case mxSINGLE_CLASS:
        componentSz = 4;
        break;
    default:
        return slowSaveMatBinary(m);
    }
    int type = CV_MAKETYPE(DepthOf[classId], channels);
    int elemSize = CV_ELEM_SIZE(type);
    int bytes = elemSize*rows*cols;
    if (rows == 0 || cols == 0) {
        mwSize rdims[] = { sizeof(int)*3, 1 };
        mxArray* r = mxCreateNumericArray(2, rdims, mxUINT8_CLASS, mxREAL);
        *(int*)mxGetData(r) = 0;
        return r;
    } else {
        mwSize rdims[] = { sizeof(int)*3 + bytes, 1 };
        mxArray* r = mxCreateNumericArray(2, rdims, mxUINT8_CLASS, mxREAL);
        const char* mp = (const char*)mxGetData(m);
        char* p = (char*)mxGetData(r);
        *(int*)p = rows; p += sizeof(int);
        *(int*)p = cols; p += sizeof(int);
        *(int*)p = type; p += sizeof(int);
        switch (componentSz) {
        case 1: {
            uint8_t* mpp = (uint8_t*)mp;
            for (int y = 0; y < rows; ++y) {
                for (int x = 0; x < cols; ++x) {
                    for (int c = 0; c < channels; ++c) {
                        *(uint8_t*)p =
                            *(mpp + y + x*rows + c*rows*cols);
                        p += 1;
                    }
                }
            }
            break;
        }
        case 4: {
            uint32_t* mpp = (uint32_t*)mp;
            for (int y = 0; y < rows; ++y) {
                for (int x = 0; x < cols; ++x) {
                    for (int c = 0; c < channels; ++c) {
                        *(uint32_t*)p =
                            *(mpp + y + x*rows + c*rows*cols);
                        p += 4;
                    }
                }
            }
            break;
        }
        default:
            mxDestroyArray(r);
            slowSaveMatBinary(m);
            return NULL;
        } 
        return r;
    }
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
    plhs[0] = saveMatBinary(prhs[0]);
}