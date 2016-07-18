// reviewed by hwb, 3.4.2016

#ifndef COLOR_VOLUME_H_
#define COLOR_VOLUME_H_

#include "../Cuda/containers/device_array.hpp"
#include <Eigen/Core>

struct TsdfVolume;

/** \brief ColorVolume class
  * \author Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
  */
class ColorVolume
{
public:
  /** \brief Constructor
    * \param[in] tsdf tsdf volume to get parameters from
    * \param[in] max_weight max weight for running average. Can be less than 255. Negative means default.
    */
  ColorVolume(const TsdfVolume& tsdf);

  /** \brief Desctructor */
  ~ColorVolume();

  /** \brief Resets color volume to uninitialized state */
  void reset();

  /** \brief Returns container with color volume in GPU memory */
  DeviceArray2D<int> data() const;

private:
  /** \brief Volume resolution */
  Eigen::Vector3i resolution_;

  /** \brief Volume size in meters */
  Eigen::Vector3f volume_size_;

  /** \brief color volume data */
  DeviceArray2D<int> color_volume_;
};

#endif /* COLOR_VOLUME_H_ */
