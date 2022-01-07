typedef double highprecision;
int timesteps=500;
const int dimX=512,dimY=512;
const int unitx=16,unity=16,unitdimX=dimX/unitx,unitdimY=dimY/unity,uxd2=unitx/2,uxd2s1=uxd2-1,uxs1=unitx-1,uys1=unity-1,dimXd2=dimX/2,unitNums=unitdimX*unitdimY;
const highprecision coefm=20.0,coefk=2.0,coefl=5.0,dvol=0.040,dvap=0.002,dsur=64,dgrb=1.6,dx=0.5,dy=0.5,dxdy=dx*dy,dtime=1.0e-4;
float R1=20,R2=R1/2;
float Ry1=240,Ry2=Ry1+R1+R2,Rx1=dimX/2+8;
const highprecision threshold=0.99;
const highprecision ratio=1.0;
