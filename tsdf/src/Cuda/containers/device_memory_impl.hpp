// reviewed by hwb, 3.4.2016
// 
#ifndef DEVICE_MEMORY_IMPL_HPP_
#define DEVICE_MEMORY_IMPL_HPP_

/////////////////////  Inline implementations of DeviceMemory ///////////////////
template<class T> inline       T* DeviceMemory::ptr()       { return (      T*)data_; }
template<class T> inline const T* DeviceMemory::ptr() const { return (const T*)data_; }
                        
template <class U> inline DeviceMemory::operator PtrSz<U>() const
{
    PtrSz<U> result;
    result.data = (U*)ptr<U>();
    result.size = sizeBytes_/sizeof(U);
    return result; 
}

/////////////////////  Inline implementations of DeviceMemory2D ///////////////////               
template<class T>        T* DeviceMemory2D::ptr(int y_arg)       { return (      T*)((      char*)data_ + y_arg * step_); }
template<class T>  const T* DeviceMemory2D::ptr(int y_arg) const { return (const T*)((const char*)data_ + y_arg * step_); }
  
template <class U> DeviceMemory2D::operator PtrStep<U>() const
{
    PtrStep<U> result;
    result.data = (U*)ptr<U>();
    result.step = step_;
    return result;
}

template <class U> DeviceMemory2D::operator PtrStepSz<U>() const
{
    PtrStepSz<U> result;
    result.data = (U*)ptr<U>();
    result.step = step_;
    result.cols = colsBytes_/sizeof(U);
    result.rows = rows_;
    return result;
}

#endif /* DEVICE_MEMORY_IMPL_HPP_ */

