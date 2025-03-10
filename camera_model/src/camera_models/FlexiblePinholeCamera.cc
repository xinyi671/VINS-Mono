// FlexiblePinholeCamera.cc
#include "camodocal/camera_models/FlexiblePinholeCamera.h"

#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camodocal/gpl/gpl.h"

namespace camodocal
{

FlexiblePinholeCamera::Parameters::Parameters()
 : Camera::Parameters(FLEXIBLE_PINHOLE) // Need to add FLEXIBLE_PINHOLE to enum
 , m_k1(0.0)
 , m_k2(0.0)
 , m_p1(0.0)
 , m_p2(0.0)
 , m_k3(0.0)
 , m_k4(0.0)
 , m_k5(0.0)
 , m_k6(0.0)
 , m_fx(0.0)
 , m_fy(0.0)
 , m_cx(0.0)
 , m_cy(0.0)
 , m_numDistortionParams(4)
{

}

FlexiblePinholeCamera::Parameters::Parameters(const std::string& cameraName,
                                      int w, int h,
                                      double k1, double k2,
                                      double p1, double p2,
                                      double k3, double k4,
                                      double k5, double k6,
                                      double fx, double fy,
                                      double cx, double cy,
                                      int numDistortionParams)
 : Camera::Parameters(FLEXIBLE_PINHOLE, cameraName, w, h)
 , m_k1(k1)
 , m_k2(k2)
 , m_p1(p1)
 , m_p2(p2)
 , m_k3(k3)
 , m_k4(k4)
 , m_k5(k5)
 , m_k6(k6)
 , m_fx(fx)
 , m_fy(fy)
 , m_cx(cx)
 , m_cy(cy)
 , m_numDistortionParams(numDistortionParams)
{
}

// Add all accessors
double& FlexiblePinholeCamera::Parameters::k1(void) { return m_k1; }
double& FlexiblePinholeCamera::Parameters::k2(void) { return m_k2; }
double& FlexiblePinholeCamera::Parameters::p1(void) { return m_p1; }
double& FlexiblePinholeCamera::Parameters::p2(void) { return m_p2; }
double& FlexiblePinholeCamera::Parameters::k3(void) { return m_k3; }
double& FlexiblePinholeCamera::Parameters::k4(void) { return m_k4; }
double& FlexiblePinholeCamera::Parameters::k5(void) { return m_k5; }
double& FlexiblePinholeCamera::Parameters::k6(void) { return m_k6; }
double& FlexiblePinholeCamera::Parameters::fx(void) { return m_fx; }
double& FlexiblePinholeCamera::Parameters::fy(void) { return m_fy; }
double& FlexiblePinholeCamera::Parameters::cx(void) { return m_cx; }
double& FlexiblePinholeCamera::Parameters::cy(void) { return m_cy; }
int& FlexiblePinholeCamera::Parameters::numDistortionParams(void) { return m_numDistortionParams; }

double FlexiblePinholeCamera::Parameters::k1(void) const { return m_k1; }
double FlexiblePinholeCamera::Parameters::k2(void) const { return m_k2; }
double FlexiblePinholeCamera::Parameters::p1(void) const { return m_p1; }
double FlexiblePinholeCamera::Parameters::p2(void) const { return m_p2; }
double FlexiblePinholeCamera::Parameters::k3(void) const { return m_k3; }
double FlexiblePinholeCamera::Parameters::k4(void) const { return m_k4; }
double FlexiblePinholeCamera::Parameters::k5(void) const { return m_k5; }
double FlexiblePinholeCamera::Parameters::k6(void) const { return m_k6; }
double FlexiblePinholeCamera::Parameters::fx(void) const { return m_fx; }
double FlexiblePinholeCamera::Parameters::fy(void) const { return m_fy; }
double FlexiblePinholeCamera::Parameters::cx(void) const { return m_cx; }
double FlexiblePinholeCamera::Parameters::cy(void) const { return m_cy; }
int FlexiblePinholeCamera::Parameters::numDistortionParams(void) const { return m_numDistortionParams; }

bool
FlexiblePinholeCamera::Parameters::readFromYamlFile(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        return false;
    }

    if (!fs["model_type"].isNone())
    {
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if (sModelType.compare("FLEXIBLE_PINHOLE") != 0)
        {
            return false;
        }
    }

    m_modelType = FLEXIBLE_PINHOLE;
    fs["camera_name"] >> m_cameraName;
    m_imageWidth = static_cast<int>(fs["image_width"]);
    m_imageHeight = static_cast<int>(fs["image_height"]);
    
    // Read number of distortion parameters
    if (!fs["num_distortion_params"].isNone())
    {
        m_numDistortionParams = static_cast<int>(fs["num_distortion_params"]);
    }
    else
    {
        // Default to 4 distortion parameters
        m_numDistortionParams = 4;
    }
    
    // Ensure parameter count is valid
    if (m_numDistortionParams < 4 || m_numDistortionParams > 8)
    {
        std::cerr << "Invalid number of distortion parameters: " << m_numDistortionParams 
                  << ". Must be between 4 and 8. Setting to 4." << std::endl;
        m_numDistortionParams = 4;
    }

    cv::FileNode n = fs["distortion_parameters"];
    m_k1 = static_cast<double>(n["k1"]);
    m_k2 = static_cast<double>(n["k2"]);
    m_p1 = static_cast<double>(n["p1"]);
    m_p2 = static_cast<double>(n["p2"]);
    
    // Read additional parameters based on the specified count
    if (m_numDistortionParams >= 5 && !n["k3"].isNone())
    {
        m_k3 = static_cast<double>(n["k3"]);
    }
    
    if (m_numDistortionParams >= 6 && !n["k4"].isNone())
    {
        m_k4 = static_cast<double>(n["k4"]);
    }
    
    if (m_numDistortionParams >= 7 && !n["k5"].isNone())
    {
        m_k5 = static_cast<double>(n["k5"]);
    }
    
    if (m_numDistortionParams >= 8 && !n["k6"].isNone())
    {
        m_k6 = static_cast<double>(n["k6"]);
    }

    n = fs["projection_parameters"];
    m_fx = static_cast<double>(n["fx"]);
    m_fy = static_cast<double>(n["fy"]);
    m_cx = static_cast<double>(n["cx"]);
    m_cy = static_cast<double>(n["cy"]);

    return true;
}

void
FlexiblePinholeCamera::Parameters::writeToYamlFile(const std::string& filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    fs << "model_type" << "FLEXIBLE_PINHOLE";
    fs << "camera_name" << m_cameraName;
    fs << "image_width" << m_imageWidth;
    fs << "image_height" << m_imageHeight;
    fs << "num_distortion_params" << m_numDistortionParams;

    // Write distortion parameters
    fs << "distortion_parameters";
    fs << "{" << "k1" << m_k1
              << "k2" << m_k2
              << "p1" << m_p1
              << "p2" << m_p2;
              
    // Write additional parameters based on the specified count
    if (m_numDistortionParams >= 5)
        fs << "k3" << m_k3;
    if (m_numDistortionParams >= 6)
        fs << "k4" << m_k4;
    if (m_numDistortionParams >= 7)
        fs << "k5" << m_k5;
    if (m_numDistortionParams >= 8)
        fs << "k6" << m_k6;
    
    fs << "}";

    // projection: fx, fy, cx, cy
    fs << "projection_parameters";
    fs << "{" << "fx" << m_fx
              << "fy" << m_fy
              << "cx" << m_cx
              << "cy" << m_cy << "}";

    fs.release();
}

FlexiblePinholeCamera::Parameters&
FlexiblePinholeCamera::Parameters::operator=(const FlexiblePinholeCamera::Parameters& other)
{
    if (this != &other)
    {
        m_modelType = other.m_modelType;
        m_cameraName = other.m_cameraName;
        m_imageWidth = other.m_imageWidth;
        m_imageHeight = other.m_imageHeight;
        m_k1 = other.m_k1;
        m_k2 = other.m_k2;
        m_p1 = other.m_p1;
        m_p2 = other.m_p2;
        m_k3 = other.m_k3;
        m_k4 = other.m_k4;
        m_k5 = other.m_k5;
        m_k6 = other.m_k6;
        m_fx = other.m_fx;
        m_fy = other.m_fy;
        m_cx = other.m_cx;
        m_cy = other.m_cy;
        m_numDistortionParams = other.m_numDistortionParams;
    }

    return *this;
}

std::ostream&
operator<< (std::ostream& out, const FlexiblePinholeCamera::Parameters& params)
{
    out << "Camera Parameters:" << std::endl;
    out << "    model_type " << "FLEXIBLE_PINHOLE" << std::endl;
    out << "   camera_name " << params.m_cameraName << std::endl;
    out << "   image_width " << params.m_imageWidth << std::endl;
    out << "  image_height " << params.m_imageHeight << std::endl;
    out << "  num_distortion_params " << params.m_numDistortionParams << std::endl;

    // Distortion parameters
    out << "Distortion Parameters" << std::endl;
    out << "            k1 " << params.m_k1 << std::endl
        << "            k2 " << params.m_k2 << std::endl
        << "            p1 " << params.m_p1 << std::endl
        << "            p2 " << params.m_p2 << std::endl;
        
    // Output additional parameters based on the specified count
    if (params.m_numDistortionParams >= 5)
        out << "            k3 " << params.m_k3 << std::endl;
    if (params.m_numDistortionParams >= 6)
        out << "            k4 " << params.m_k4 << std::endl;
    if (params.m_numDistortionParams >= 7)
        out << "            k5 " << params.m_k5 << std::endl;
    if (params.m_numDistortionParams >= 8)
        out << "            k6 " << params.m_k6 << std::endl;

    // projection: fx, fy, cx, cy
    out << "Projection Parameters" << std::endl;
    out << "            fx " << params.m_fx << std::endl
        << "            fy " << params.m_fy << std::endl
        << "            cx " << params.m_cx << std::endl
        << "            cy " << params.m_cy << std::endl;

    return out;
}

FlexiblePinholeCamera::FlexiblePinholeCamera()
 : m_inv_K11(1.0)
 , m_inv_K13(0.0)
 , m_inv_K22(1.0)
 , m_inv_K23(0.0)
 , m_noDistortion(true)
{

}

FlexiblePinholeCamera::FlexiblePinholeCamera(const std::string& cameraName,
                             int imageWidth, int imageHeight,
                             double k1, double k2, double p1, double p2,
                             double k3, double k4, double k5, double k6,
                             double fx, double fy, double cx, double cy,
                             int numDistortionParams)
 : mParameters(cameraName, imageWidth, imageHeight,
               k1, k2, p1, p2, k3, k4, k5, k6, fx, fy, cx, cy, numDistortionParams)
{
    if ((mParameters.k1() == 0.0) &&
        (mParameters.k2() == 0.0) &&
        (mParameters.p1() == 0.0) &&
        (mParameters.p2() == 0.0) &&
        (mParameters.k3() == 0.0) &&
        (mParameters.k4() == 0.0) &&
        (mParameters.k5() == 0.0) &&
        (mParameters.k6() == 0.0))
    {
        m_noDistortion = true;
    }
    else
    {
        m_noDistortion = false;
    }

    // Inverse camera projection matrix parameters
    m_inv_K11 = 1.0 / mParameters.fx();
    m_inv_K13 = -mParameters.cx() / mParameters.fx();
    m_inv_K22 = 1.0 / mParameters.fy();
    m_inv_K23 = -mParameters.cy() / mParameters.fy();
}

FlexiblePinholeCamera::FlexiblePinholeCamera(const FlexiblePinholeCamera::Parameters& params)
 : mParameters(params)
{
    if ((mParameters.k1() == 0.0) &&
        (mParameters.k2() == 0.0) &&
        (mParameters.p1() == 0.0) &&
        (mParameters.p2() == 0.0) &&
        (mParameters.k3() == 0.0) &&
        (mParameters.k4() == 0.0) &&
        (mParameters.k5() == 0.0) &&
        (mParameters.k6() == 0.0))
    {
        m_noDistortion = true;
    }
    else
    {
        m_noDistortion = false;
    }

    // Inverse camera projection matrix parameters
    m_inv_K11 = 1.0 / mParameters.fx();
    m_inv_K13 = -mParameters.cx() / mParameters.fx();
    m_inv_K22 = 1.0 / mParameters.fy();
    m_inv_K23 = -mParameters.cy() / mParameters.fy();
}

Camera::ModelType
FlexiblePinholeCamera::modelType(void) const
{
    return mParameters.modelType();
}

const std::string&
FlexiblePinholeCamera::cameraName(void) const
{
    return mParameters.cameraName();
}

int
FlexiblePinholeCamera::imageWidth(void) const
{
    return mParameters.imageWidth();
}

int
FlexiblePinholeCamera::imageHeight(void) const
{
    return mParameters.imageHeight();
}

int 
FlexiblePinholeCamera::numDistortionParams(void) const
{
    return mParameters.numDistortionParams();
}

void
FlexiblePinholeCamera::estimateIntrinsics(const cv::Size& boardSize,
                            const std::vector< std::vector<cv::Point3f> >& objectPoints,
                            const std::vector< std::vector<cv::Point2f> >& imagePoints)
{
    // Z. Zhang, A Flexible New Technique for Camera Calibration, PAMI 2000

    Parameters params = getParameters();

    params.k1() = 0.0;
    params.k2() = 0.0;
    params.p1() = 0.0;
    params.p2() = 0.0;
    params.k3() = 0.0;
    params.k4() = 0.0;
    params.k5() = 0.0;
    params.k6() = 0.0;

    double cx = params.imageWidth() / 2.0;
    double cy = params.imageHeight() / 2.0;
    params.cx() = cx;
    params.cy() = cy;

    size_t nImages = imagePoints.size();

    cv::Mat A(nImages * 2, 2, CV_64F);
    cv::Mat b(nImages * 2, 1, CV_64F);

    for (size_t i = 0; i < nImages; ++i)
    {
        const std::vector<cv::Point3f>& oPoints = objectPoints.at(i);

        std::vector<cv::Point2f> M(oPoints.size());
        for (size_t j = 0; j < M.size(); ++j)
        {
            M.at(j) = cv::Point2f(oPoints.at(j).x, oPoints.at(j).y);
        }

        cv::Mat H = cv::findHomography(M, imagePoints.at(i));

        H.at<double>(0,0) -= H.at<double>(2,0) * cx;
        H.at<double>(0,1) -= H.at<double>(2,1) * cx;
        H.at<double>(0,2) -= H.at<double>(2,2) * cx;
        H.at<double>(1,0) -= H.at<double>(2,0) * cy;
        H.at<double>(1,1) -= H.at<double>(2,1) * cy;
        H.at<double>(1,2) -= H.at<double>(2,2) * cy;

        double h[3], v[3], d1[3], d2[3];
        double n[4] = {0,0,0,0};

        for (int j = 0; j < 3; ++j)
        {
            double t0 = H.at<double>(j,0);
            double t1 = H.at<double>(j,1);
            h[j] = t0; v[j] = t1;
            d1[j] = (t0 + t1) * 0.5;
            d2[j] = (t0 - t1) * 0.5;
            n[0] += t0 * t0; n[1] += t1 * t1;
            n[2] += d1[j] * d1[j]; n[3] += d2[j] * d2[j];
        }

        for (int j = 0; j < 4; ++j)
        {
            n[j] = 1.0 / sqrt(n[j]);
        }

        for (int j = 0; j < 3; ++j)
        {
            h[j] *= n[0]; v[j] *= n[1];
            d1[j] *= n[2]; d2[j] *= n[3];
        }

        A.at<double>(i * 2, 0) = h[0] * v[0];
        A.at<double>(i * 2, 1) = h[1] * v[1];
        A.at<double>(i * 2 + 1, 0) = d1[0] * d2[0];
        A.at<double>(i * 2 + 1, 1) = d1[1] * d2[1];
        b.at<double>(i * 2, 0) = -h[2] * v[2];
        b.at<double>(i * 2 + 1, 0) = -d1[2] * d2[2];
    }

    cv::Mat f(2, 1, CV_64F);
    cv::solve(A, b, f, cv::DECOMP_NORMAL | cv::DECOMP_LU);

    params.fx() = sqrt(fabs(1.0 / f.at<double>(0)));
    params.fy() = sqrt(fabs(1.0 / f.at<double>(1)));

    setParameters(params);
}

void
FlexiblePinholeCamera::liftSphere(const Eigen::Vector2d& p, Eigen::Vector3d& P) const
{
    liftProjective(p, P);

    P.normalize();
}

void
FlexiblePinholeCamera::distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) const
{
    double k1 = mParameters.k1();
    double k2 = mParameters.k2();
    double p1 = mParameters.p1();
    double p2 = mParameters.p2();
    
    double k3 = 0.0, k4 = 0.0, k5 = 0.0, k6 = 0.0;
    if (mParameters.numDistortionParams() >= 5) k3 = mParameters.k3();
    if (mParameters.numDistortionParams() >= 6) k4 = mParameters.k4();
    if (mParameters.numDistortionParams() >= 7) k5 = mParameters.k5();
    if (mParameters.numDistortionParams() >= 8) k6 = mParameters.k6();

    double mx2_u, my2_u, mxy_u, rho2_u, rho4_u, rho6_u, rho8_u;
    
    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rho4_u = rho2_u * rho2_u;
    rho6_u = rho4_u * rho2_u;
    rho8_u = rho6_u * rho2_u;
    
    double radial_distortion = k1 * rho2_u + k2 * rho4_u;
    
    if (mParameters.numDistortionParams() >= 5) {
        radial_distortion += k3 * rho6_u;
    }
    
    if (mParameters.numDistortionParams() >= 6) {
        radial_distortion += k4 * rho8_u;
    }
    
    double x_distorted = p_u(0) * (1.0 + radial_distortion) + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u);
    double y_distorted = p_u(1) * (1.0 + radial_distortion) + p1 * (rho2_u + 2.0 * my2_u) + 2.0 * p2 * mxy_u;
    
    if (mParameters.numDistortionParams() >= 7) {
        x_distorted += k5 * rho2_u * p_u(0);
    }
    
    if (mParameters.numDistortionParams() >= 8) {
        y_distorted += k6 * rho2_u * p_u(1);
    }
    
    d_u << x_distorted - p_u(0), y_distorted - p_u(1);
}

void
FlexiblePinholeCamera::distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u,
                          Eigen::Matrix2d& J) const
{
    double k1 = mParameters.k1();
    double k2 = mParameters.k2();
    double p1 = mParameters.p1();
    double p2 = mParameters.p2();
    
    // Additional distortion parameters
    double k3 = 0.0, k4 = 0.0, k5 = 0.0, k6 = 0.0;
    if (mParameters.numDistortionParams() >= 5) k3 = mParameters.k3();
    if (mParameters.numDistortionParams() >= 6) k4 = mParameters.k4();
    if (mParameters.numDistortionParams() >= 7) k5 = mParameters.k5();
    if (mParameters.numDistortionParams() >= 8) k6 = mParameters.k6();

    double mx2_u, my2_u, mxy_u, rho2_u, rho4_u, rho6_u, rho8_u;

    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rho4_u = rho2_u * rho2_u;
    rho6_u = rho4_u * rho2_u;
    rho8_u = rho6_u * rho2_u;
    
    // Apply distortion model based on parameter count
    double rad_dist_u = k1 * rho2_u + k2 * rho4_u;
    
    if (mParameters.numDistortionParams() >= 5) {
        rad_dist_u += k3 * rho6_u;
    }
    
    if (mParameters.numDistortionParams() >= 6) {
        rad_dist_u += k4 * rho8_u;
    }
    
    // Calculate distortion vector
    double dx = p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u);
    double dy = p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
    
    if (mParameters.numDistortionParams() >= 7) {
        dx += k5 * rho2_u * p_u(0);
    }
    
    if (mParameters.numDistortionParams() >= 8) {
        dx += k6 * rho2_u * p_u(1);
    }
    
    d_u << dx, dy;
    
    // Calculate Jacobian
    double drdx = 2.0 * p_u(0) * k1 + 4.0 * p_u(0) * rho2_u * k2;
    double drdy = 2.0 * p_u(1) * k1 + 4.0 * p_u(1) * rho2_u * k2;
    
    if (mParameters.numDistortionParams() >= 5) {
        drdx += 6.0 * p_u(0) * rho4_u * k3;
        drdy += 6.0 * p_u(1) * rho4_u * k3;
    }
    
    if (mParameters.numDistortionParams() >= 6) {
        drdx += 8.0 * p_u(0) * rho6_u * k4;
        drdy += 8.0 * p_u(1) * rho6_u * k4;
    }
    
    // Base Jacobian
    double dxdx = rad_dist_u + p_u(0) * drdx + 2.0 * p1 * p_u(1) + 6.0 * p2 * p_u(0);
    double dxdy = p_u(0) * drdy + 2.0 * p1 * p_u(0) + 2.0 * p2 * p_u(1);
    double dydx = p_u(1) * drdx + 2.0 * p2 * p_u(1) + 2.0 * p1 * p_u(0);
    double dydy = rad_dist_u + p_u(1) * drdy + 2.0 * p2 * p_u(0) + 6.0 * p1 * p_u(1);
    
    // Add additional terms for k5 and k6
    if (mParameters.numDistortionParams() >= 7) {
        dxdx += k5 * (rho2_u + 2.0 * mx2_u);
        dxdy += k5 * 2.0 * p_u(0) * p_u(1);
    }
    
    if (mParameters.numDistortionParams() >= 8) {
        dxdx += k6 * 2.0 * p_u(0) * p_u(1);
        dxdy += k6 * (rho2_u + 2.0 * my2_u);
    }
    
    J << dxdx, dxdy,
         dydx, dydy;
}

void
FlexiblePinholeCamera::liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const
{
    double mx_d, my_d, mx_u, my_u;

    mx_d = m_inv_K11 * p(0) + m_inv_K13;
    my_d = m_inv_K22 * p(1) + m_inv_K23;

    if (m_noDistortion)
    {
        mx_u = mx_d;
        my_u = my_d;
    }
    else
    {
        std::vector<double> parameterVec;
        writeParameters(parameterVec);
        
        int numDistCoeffs = parameterVec.size() - 4; 
        
        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = mParameters.fx();
        K.at<double>(1, 1) = mParameters.fy();
        K.at<double>(0, 2) = mParameters.cx();
        K.at<double>(1, 2) = mParameters.cy();
        
        cv::Mat distCoeffs = cv::Mat::zeros(numDistCoeffs, 1, CV_64F);
        for (int i = 0; i < numDistCoeffs; i++) {
            distCoeffs.at<double>(i) = parameterVec[i];
        }
        
        std::vector<cv::Point2f> src(1);
        std::vector<cv::Point2f> dst(1);
        
        src[0].x = p(0);
        src[0].y = p(1);
        
        cv::undistortPoints(src, dst, K, distCoeffs);
        
        mx_u = dst[0].x;
        my_u = dst[0].y;
    }

    P << mx_u, my_u, 1.0;
}

void
FlexiblePinholeCamera::spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p) const
{
    Eigen::Vector2d p_u, p_d;

    // Project point to the normalized plane
    p_u << P(0) / P(2), P(1) / P(2);

    if (m_noDistortion)
    {
        p_d = p_u;
    }
    else
    {
        // Apply distortion
        Eigen::Vector2d d_u;
        distortion(p_u, d_u);
        p_d = p_u + d_u;
    }

    // Apply projection matrix
    p << mParameters.fx() * p_d(0) + mParameters.cx(),
         mParameters.fy() * p_d(1) + mParameters.cy();
}

void
FlexiblePinholeCamera::undistToPlane(const Eigen::Vector2d& p_u, Eigen::Vector2d& p) const
{
    Eigen::Vector2d p_d;

    if (m_noDistortion)
    {
        p_d = p_u;
    }
    else
    {
        // Apply distortion
        Eigen::Vector2d d_u;
        distortion(p_u, d_u);
        p_d = p_u + d_u;
    }

    // Apply projection matrix
    p << mParameters.fx() * p_d(0) + mParameters.cx(),
         mParameters.fy() * p_d(1) + mParameters.cy();
}

void
FlexiblePinholeCamera::initUndistortMap(cv::Mat& map1, cv::Mat& map2, double fScale) const
{
    cv::Size imageSize(mParameters.imageWidth(), mParameters.imageHeight());

    cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

    for (int v = 0; v < imageSize.height; ++v)
    {
        for (int u = 0; u < imageSize.width; ++u)
        {
            double mx_u = m_inv_K11 / fScale * u + m_inv_K13 / fScale;
            double my_u = m_inv_K22 / fScale * v + m_inv_K23 / fScale;

            Eigen::Vector3d P;
            P << mx_u, my_u, 1.0;

            Eigen::Vector2d p;
            spaceToPlane(P, p);

            mapX.at<float>(v,u) = p(0);
            mapY.at<float>(v,u) = p(1);
        }
    }

    cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);
}

cv::Mat
FlexiblePinholeCamera::initUndistortRectifyMap(cv::Mat& map1, cv::Mat& map2,
                                       float fx, float fy,
                                       cv::Size imageSize,
                                       float cx, float cy,
                                       cv::Mat rmat) const
{
    if (imageSize == cv::Size(0, 0))
    {
        imageSize = cv::Size(mParameters.imageWidth(), mParameters.imageHeight());
    }

    cv::Mat mapX = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);

    Eigen::Matrix3f R, R_inv;
    cv::cv2eigen(rmat, R);
    R_inv = R.inverse();

    // assume no skew
    Eigen::Matrix3f K_rect;

    if (cx == -1.0f || cy == -1.0f)
    {
        K_rect << fx, 0, imageSize.width / 2,
                  0, fy, imageSize.height / 2,
                  0, 0, 1;
    }
    else
    {
        K_rect << fx, 0, cx,
                  0, fy, cy,
                  0, 0, 1;
    }

    if (fx == -1.0f || fy == -1.0f)
    {
        K_rect(0,0) = mParameters.fx();
        K_rect(1,1) = mParameters.fy();
    }

    Eigen::Matrix3f K_rect_inv = K_rect.inverse();

    for (int v = 0; v < imageSize.height; ++v)
    {
        for (int u = 0; u < imageSize.width; ++u)
        {
            Eigen::Vector3f xo;
            xo << u, v, 1;

            Eigen::Vector3f uo = R_inv * K_rect_inv * xo;

            Eigen::Vector2d p;
            spaceToPlane(uo.cast<double>(), p);

            mapX.at<float>(v,u) = p(0);
            mapY.at<float>(v,u) = p(1);
        }
    }

    cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);

    cv::Mat K_rect_cv;
    cv::eigen2cv(K_rect, K_rect_cv);
    return K_rect_cv;
}

int
FlexiblePinholeCamera::parameterCount(void) const
{
    return 4 + mParameters.numDistortionParams(); // 4 intrinsic parameters + n distortion parameters
}

const FlexiblePinholeCamera::Parameters&
FlexiblePinholeCamera::getParameters(void) const
{
    return mParameters;
}


void
FlexiblePinholeCamera::setParameters(const FlexiblePinholeCamera::Parameters& parameters)
{
    mParameters = parameters;
    int numDistParams = mParameters.numDistortionParams();
    std::cout << "distortion: " << numDistParams << " " <<
    mParameters.k1() << " " << mParameters.k2() << " " << mParameters.p1() << " " << 
    mParameters.p2() << " " << mParameters.k3() << " " << mParameters.k4() << " " << 
    mParameters.k5() << " " << mParameters.k6() << std::endl;

    
    // Initialize to true, assuming no distortion
    m_noDistortion = true;
    
    // Check basic 4 distortion parameters
    if (mParameters.k1() != 0.0 || mParameters.k2() != 0.0 ||
        mParameters.p1() != 0.0 || mParameters.p2() != 0.0)
    {
        m_noDistortion = false;
    }
    
    // Check additional parameters based on distortion parameter count
    if (numDistParams >= 5 && mParameters.k3() != 0.0)
        m_noDistortion = false;
    if (numDistParams >= 6 && mParameters.k4() != 0.0)
        m_noDistortion = false;
    if (numDistParams >= 7 && mParameters.k5() != 0.0)
        m_noDistortion = false;
    if (numDistParams >= 8 && mParameters.k6() != 0.0)
        m_noDistortion = false;
    
    m_inv_K11 = 1.0 / mParameters.fx();
    m_inv_K13 = -mParameters.cx() / mParameters.fx();
    m_inv_K22 = 1.0 / mParameters.fy();
    m_inv_K23 = -mParameters.cy() / mParameters.fy();
}

void
FlexiblePinholeCamera::readParameters(const std::vector<double>& parameterVec)
{
    int numDistParams = mParameters.numDistortionParams();
    
    // Validate parameter count
    if ((int)parameterVec.size() != 4 + numDistParams)
    {
        return;
    }

    Parameters params = getParameters();

    // Read distortion parameters
    params.k1() = parameterVec.at(0);
    params.k2() = parameterVec.at(1);
    params.p1() = parameterVec.at(2);
    params.p2() = parameterVec.at(3);
    
    // Read additional distortion parameters
    if (numDistParams >= 5) params.k3() = parameterVec.at(4);
    if (numDistParams >= 6) params.k4() = parameterVec.at(5);
    if (numDistParams >= 7) params.k5() = parameterVec.at(6);
    if (numDistParams >= 8) params.k6() = parameterVec.at(7);

    // Read intrinsic parameters
    params.fx() = parameterVec.at(numDistParams);
    params.fy() = parameterVec.at(numDistParams + 1);
    params.cx() = parameterVec.at(numDistParams + 2);
    params.cy() = parameterVec.at(numDistParams + 3);

    setParameters(params);
}

void
FlexiblePinholeCamera::writeParameters(std::vector<double>& parameterVec) const
{
    int numDistParams = mParameters.numDistortionParams();
    parameterVec.resize(4 + numDistParams);
    
    // Write basic distortion parameters
    parameterVec.at(0) = mParameters.k1();
    parameterVec.at(1) = mParameters.k2();
    parameterVec.at(2) = mParameters.p1();
    parameterVec.at(3) = mParameters.p2();
    
    // Write additional distortion parameters
    if (numDistParams >= 5) parameterVec.at(4) = mParameters.k3();
    if (numDistParams >= 6) parameterVec.at(5) = mParameters.k4();
    if (numDistParams >= 7) parameterVec.at(6) = mParameters.k5();
    if (numDistParams >= 8) parameterVec.at(7) = mParameters.k6();
    
    // Write intrinsic parameters
    parameterVec.at(numDistParams) = mParameters.fx();
    parameterVec.at(numDistParams + 1) = mParameters.fy();
    parameterVec.at(numDistParams + 2) = mParameters.cx();
    parameterVec.at(numDistParams + 3) = mParameters.cy();
}

void
FlexiblePinholeCamera::writeParametersToYamlFile(const std::string& filename) const
{
    mParameters.writeToYamlFile(filename);
}

std::string
FlexiblePinholeCamera::parametersToString(void) const
{
    std::ostringstream oss;
    oss << mParameters;

    return oss.str();
}

}