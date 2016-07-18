// reviewed by hwb, 3.4.2016

#include <algorithm>
#include <Eigen/Core>
#include "ColorVolume.h"
#include "TSDFVolume.h"

ColorVolume::ColorVolume(const TsdfVolume& tsdf)
 : resolution_(tsdf.getResolution()),
   volume_size_(tsdf.getSize())
{
    int volume_x = resolution_(0);
    int volume_y = resolution_(1);
    int volume_z = resolution_(2);
    color_volume_.create(volume_y * volume_z, volume_x);
    reset();
}

ColorVolume::~ColorVolume()
{

}

void ColorVolume::reset()
{
    initColorVolume(color_volume_);
}

DeviceArray2D<int> ColorVolume::data() const
{
    return color_volume_;
}

