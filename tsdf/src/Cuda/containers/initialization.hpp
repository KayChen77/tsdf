// reviewed by hwb, 3.4.2016

#ifndef INITIALISATION_HPP_
#define INITIALISATION_HPP_

#include <string>

/** \brief Returns number of Cuda device. */
int getCudaEnabledDeviceCount();

/** \brief Sets active device to work with. */
void setDevice(int device);

/** \brief Return devuce name for gived device. */
std::string getDeviceName(int device);

/** \brief Prints infromatoin about given cuda deivce or about all deivces
 *  \param deivce: if < 0 prints info for all devices, otherwise the function interpets is as device id.
 */
void printCudaDeviceInfo(int device = -1);

/** \brief Prints infromatoin about given cuda deivce or about all deivces
 *  \param deivce: if < 0 prints info for all devices, otherwise the function interpets is as device id.
 */
void printShortCudaDeviceInfo(int device = -1);

/** \brief Returns true if pre-Fermi generaton GPU.
  * \param device: device id to check, if < 0 checks current device.
  */
bool checkIfPreFermiGPU(int device = -1);

/** \brief Error handler. All GPU functions call this to report an error. For internal use only */
void error(const char *error_string, const char *file, const int line, const char *func = "");

#endif /* INITIALISATION_HPP_ */
