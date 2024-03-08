import numpy as np

class Conversion():
    
    def __build_time_wavelet(self):

        ns = int(4.0*np.pi/self.__fmax/self.__dt)

        self.__wavelet = np.zeros(ns)
 
        fc = self.__fmax / (3.0 * np.sqrt(np.pi))
        
        mid = int(0.5*ns)
        for i in range(-mid, mid):
            arg = pow(i*self.__dt*fc*np.pi, 2.0)
            self.__wavelet[i + mid] = (1.0 - 2.0*arg)*np.exp(-arg) 

    def __build_depth_wavelet(self):

        ns = int(0.25*np.pi/self.__fmax/self.__dt)

        self.__wavelet = np.zeros(ns) 

        mid = int(0.5*ns)
        for i in range(-mid, mid):
            arg = pow(i*self.__dt*self.__fmax*np.pi**2, 2.0)
            self.__wavelet[i + mid] = (1.0 - 2.0*arg)*np.exp(-arg) 

    def __build_depth_amplitudes(self):

        reflectivity = np.zeros_like(self.__model)    

        reflectivity[:-1] = (self.__model[1:] - self.__model[:-1]) / (self.__model[1:] + self.__model[:-1]) 

        transmission = np.zeros_like(reflectivity)  

        self.__amp_depth = np.zeros_like(reflectivity)

        self.__amp_depth[0] = reflectivity[0]         
        transmission[0] = 1 - reflectivity[0]     

        for i in range(1, len(reflectivity)):   
            self.__amp_depth[i] = transmission[i-1] * reflectivity[i]         
            transmission[i] = transmission[i-1] * (1 - reflectivity[i]) 

            for j in range(i, 0, -1):  
                self.__amp_depth[i] *= (1 - reflectivity[i - j])    

    def __build_time_amplitudes(self):
  
        t = np.zeros(self.__nx, dtype = int)
        self.__amp_time = np.zeros((self.__nt, self.__nx))
    
        for i in range(self.__nz):
            t += np.array(2 * (self.__dh / self.__model[i]) / self.__dt, dtype = int)
            for j in range(self.__nx):
                if t[j] < self.__nt:
                    self.__amp_time[t[j],j] = self.__amp_depth[i,j]

    def model_to_time_section(self, velocity_model:np.ndarray, model_spacing:float, total_time:float, time_spacing:float, max_frequency:float) -> np.ndarray:
        
        self.__dt = time_spacing
        self.__dh = model_spacing
        self.__fmax = max_frequency
        self.__model = velocity_model

        self.__nt = int(total_time / time_spacing) + 1

        self.__nz, self.__nx = np.shape(self.__model)

        self.__build_time_wavelet()
        self.__build_depth_amplitudes()
        self.__build_time_amplitudes()

        self.__time_section = np.zeros_like(self.__amp_time)
        for j in range(self.__nx):
            self.__time_section[:,j] = np.convolve(self.__wavelet, self.__amp_time[:,j], "same")
        
        return self.__time_section
    
    def model_to_depth_section(self, velocity_model:np.ndarray, model_spacing:float, total_time:float, time_spacing:float, max_frequency:float) -> np.ndarray:
        self.__dt = time_spacing
        self.__dh = model_spacing
        self.__fmax = max_frequency
        self.__model = velocity_model

        self.__nt = int(total_time / time_spacing) + 1

        self.__nz, self.__nx = np.shape(self.__model)

        self.__build_depth_wavelet()
        self.__build_depth_amplitudes()

        self.__depth_section = np.zeros_like(self.__amp_depth)
        for j in range(self.__nx):
            self.__depth_section[:,j] = np.convolve(self.__wavelet, self.__amp_depth[:,j], "same")

        return self.__depth_section

    def time_section_to_depth_section():
        pass

    def depth_section_to_time_section():
        pass

