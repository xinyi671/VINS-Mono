// FlexiblePinholeCamera.h
#ifndef FLEXIBLE_PINHOLE_CAMERA_H
#define FLEXIBLE_PINHOLE_CAMERA_H

#include <opencv2/core/core.hpp>
#include <string>

#include "ceres/rotation.h"
#include "Camera.h"
#include <iostream>


namespace camodocal
{

class FlexiblePinholeCamera: public Camera
{
public:
    class Parameters: public Camera::Parameters
    {
    public:
        Parameters();
        Parameters(const std::string& cameraName,
                   int w, int h,
                   double k1, double k2, double p1, double p2,
                   double k3 = 0.0, double k4 = 0.0, double k5 = 0.0, double k6 = 0.0,
                   double fx = 0.0, double fy = 0.0, double cx = 0.0, double cy = 0.0,
                   int numDistortionParams = 4);

        double& k1(void);
        double& k2(void);
        double& p1(void);
        double& p2(void);
        double& k3(void);
        double& k4(void);
        double& k5(void);
        double& k6(void);
        double& fx(void);
        double& fy(void);
        double& cx(void);
        double& cy(void);
        int& numDistortionParams(void);

        double k1(void) const;
        double k2(void) const;
        double p1(void) const;
        double p2(void) const;
        double k3(void) const;
        double k4(void) const;
        double k5(void) const;
        double k6(void) const;
        double fx(void) const;
        double fy(void) const;
        double cx(void) const;
        double cy(void) const;
        int numDistortionParams(void) const;

        bool readFromYamlFile(const std::string& filename);
        void writeToYamlFile(const std::string& filename) const;

        Parameters& operator=(const Parameters& other);
        friend std::ostream& operator<< (std::ostream& out, const Parameters& params);

    private:
        double m_k1;
        double m_k2;
        double m_p1;
        double m_p2;
        double m_k3;
        double m_k4;
        double m_k5;
        double m_k6;
        double m_fx;
        double m_fy;
        double m_cx;
        double m_cy;
        int m_numDistortionParams; // 4, 5, 6, 7
    };

    FlexiblePinholeCamera();

    /**
    * \brief Constructor from the projection model parameters
    */
    FlexiblePinholeCamera(const std::string& cameraName,
                  int imageWidth, int imageHeight,
                  double k1, double k2, double p1, double p2,
                  double k3, double k4, double k5, double k6,
                  double fx, double fy, double cx, double cy,
                  int numDistortionParams = 4);
    /**
    * \brief Constructor from the projection model parameters
    */
    FlexiblePinholeCamera(const Parameters& params);

    Camera::ModelType modelType(void) const;
    const std::string& cameraName(void) const;
    int imageWidth(void) const;
    int imageHeight(void) const;
    int numDistortionParams(void) const;

    void estimateIntrinsics(const cv::Size& boardSize,
                            const std::vector< std::vector<cv::Point3f> >& objectPoints,
                            const std::vector< std::vector<cv::Point2f> >& imagePoints);

    // Lift points from the image plane to the sphere
    virtual void liftSphere(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    //%output P

    // Lift points from the image plane to the projective space
    void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    //%output P

    // Projects 3D points to the image plane (Pi function)
    void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p) const;
    //%output p

    void undistToPlane(const Eigen::Vector2d& p_u, Eigen::Vector2d& p) const;
    //%output p

    template <typename T>
    static void spaceToPlane(const T* const params,
                             const T* const q, const T* const t,
                             const Eigen::Matrix<T, 3, 1>& P,
                             Eigen::Matrix<T, 2, 1>& p,
                             int numDistortionParams);

    void distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) const;
    void distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u,
                    Eigen::Matrix2d& J) const;

    void initUndistortMap(cv::Mat& map1, cv::Mat& map2, double fScale = 1.0) const;
    cv::Mat initUndistortRectifyMap(cv::Mat& map1, cv::Mat& map2,
                                    float fx = -1.0f, float fy = -1.0f,
                                    cv::Size imageSize = cv::Size(0, 0),
                                    float cx = -1.0f, float cy = -1.0f,
                                    cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F)) const;

    int parameterCount(void) const;

    const Parameters& getParameters(void) const;
    void setParameters(const Parameters& parameters);

    void readParameters(const std::vector<double>& parameterVec);
    void writeParameters(std::vector<double>& parameterVec) const;

    void writeParametersToYamlFile(const std::string& filename) const;

    std::string parametersToString(void) const;

private:
    Parameters mParameters;

    double m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23;
    bool m_noDistortion;
};

typedef boost::shared_ptr<FlexiblePinholeCamera> FlexiblePinholeCameraPtr;
typedef boost::shared_ptr<const FlexiblePinholeCamera> FlexiblePinholeCameraConstPtr;

template <typename T>
void
FlexiblePinholeCamera::spaceToPlane(const T* const params,
                            const T* const q, const T* const t,
                            const Eigen::Matrix<T, 3, 1>& P,
                            Eigen::Matrix<T, 2, 1>& p,
                            int numDistortionParams)
{
    T P_w[3];
    P_w[0] = T(P(0));
    P_w[1] = T(P(1));
    P_w[2] = T(P(2));

    // Convert quaternion from Eigen convention (x, y, z, w)
    // to Ceres convention (w, x, y, z)
    T q_ceres[4] = {q[3], q[0], q[1], q[2]};

    T P_c[3];
    ceres::QuaternionRotatePoint(q_ceres, P_w, P_c);

    P_c[0] += t[0];
    P_c[1] += t[1];
    P_c[2] += t[2];

    // project 3D object point to the image plane
    T k1 = params[0];
    T k2 = params[1];
    T p1 = params[2];
    T p2 = params[3];
    
    // extra distortion parameters
    T k3 = numDistortionParams >= 5 ? params[4] : T(0);
    T k4 = numDistortionParams >= 6 ? params[5] : T(0);
    T k5 = numDistortionParams >= 7 ? params[6] : T(0);
    T k6 = numDistortionParams >= 8 ? params[7] : T(0);
    
    // intrinsics
    int offset = numDistortionParams;
    T fx = params[offset];
    T fy = params[offset + 1];
    T alpha = T(0); // skew parameter, not used here
    T cx = params[offset + 2];
    T cy = params[offset + 3];

    // Transform to model plane
    T u = P_c[0] / P_c[2];
    T v = P_c[1] / P_c[2];

    T rho2 = u * u + v * v;
    T rho4 = rho2 * rho2;
    T rho6 = rho4 * rho2;
    T rho8 = rho6 * rho2;
    
    // choose the right distortion model
    T radial = T(1.0);
    
    if (numDistortionParams >= 4) {
        radial += k1 * rho2 + k2 * rho4;
    }
    
    if (numDistortionParams >= 5) {
        radial += k3 * rho6;
    }
    
    if (numDistortionParams >= 6) {
        radial += k4 * rho8;
    }
    
    T dx = T(2.0) * p1 * u * v + p2 * (rho2 + T(2.0) * u * u);
    
    if (numDistortionParams >= 7) {
        dx += k5 * rho2 * u;
    }
    
    if (numDistortionParams >= 8) {
        dx += k6 * rho2 * v;
    }
    
    T dy = p1 * (rho2 + T(2.0) * v * v) + T(2.0) * p2 * u * v;

    u = radial * u + dx;
    v = radial * v + dy;
    p(0) = fx * (u + alpha * v) + cx;
    p(1) = fy * v + cy;
}

}

#endif