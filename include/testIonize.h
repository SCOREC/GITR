struct ionize { 
  ...//Accept inputs including Temp, Dens, particles, dt etc.
  CUDA_CALLABLE_MEMBER_DEVICE 
  void operator()(std::size_t indx) const { 
    double ionTemperature = getIonTemperature()
    double ionDensity = getIonDensity()
    double P = exp(-dt/ionTemperature);
    double P1 = 1.0-P;
    double r1 = rand()
    double chargeAdd = 0.0;
    if(r1 <= P1){
      chargeAdd = 1.0;}
    particles->charge[indx] = particles->charge[indx]+chargeAdd;
  }
