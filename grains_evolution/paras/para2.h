typedef float highprecision;
int timesteps=500;
const int dimX=512,dimY=512;
const int unitx=16,unity=16,unitdimX=dimX/unitx,unitdimY=dimY/unity,uxd2=unitx/2,uxd2s1=uxd2-1,uxs1=unitx-1,uys1=unity-1,dimXd2=dimX/2,unitNums=unitdimX*unitdimY;
const highprecision dtime=0.005,mobil=10.0,dx=0.5,dy=0.5,dxdy=dx*dy,grcoef=0.1;
float R=50;
float Ry=dimY/2,Rx=dimX/2;
const highprecision threshold=0.99;
const highprecision ratio=1.0;
