/**
 * @file bincvmat.cpp
 * @brief mex interface for load opencv binary mat file
 * @ingroup core
 */
#include "mexopencv.hpp"
using namespace std;
using namespace cv;

#include <fstream>

static bool readMatBinary(ifstream& ifs, Mat& in_mat)
{
    if(!ifs.is_open()){
        return false;
    }

    int rows, cols, type;
    ifs.read((char*)(&rows), sizeof(int));
    if(rows==0){
        return true;
    }
    ifs.read((char*)(&cols), sizeof(int));
    ifs.read((char*)(&type), sizeof(int));

    in_mat.release();
    in_mat.create(rows, cols, type);
    ifs.read((char*)(in_mat.data), in_mat.elemSize() * in_mat.total());

    return true;
}

static bool LoadMatBinary(const string& filename, Mat& output)
{
    ifstream ifs(filename, ios::binary);
    return readMatBinary(ifs, output);
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

    // Argument vector
    vector<MxArray> rhs(prhs, prhs+nrhs);

    // Process
    string filename(rhs[0].toString());
    Mat img;
    if (LoadMatBinary(filename, img) == false) {
        mexErrMsgIdAndTxt("mexopencv:error", "bincvmat failed");
    }
    plhs[0] = MxArray(img);
}
