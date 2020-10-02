#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>

#include "func.hpp"

template<typename T>
auto generate_container(size_t size, const T& val){
  //gotta use pointer arrays for correct numa placement of memory
  T* tmp = new T[size];
#pragma omp parallel for default(none) shared(tmp) firstprivate(size,val) schedule(static)
  for(size_t i=0; i<size; ++i){
    tmp[i] = val;
  }

  return tmp;
}

int main(){
    //boundaries
    constexpr double boundary_val_l = 0.0;
    constexpr double boundary_val_r = 0.0;

    // grid infomations
    constexpr double interval_w = 1.0;
    constexpr int num_split = 100000;
    constexpr double h = interval_w/static_cast<double>(num_split);
    constexpr double hsq = h*h;

    // deviation where we stop
    constexpr double d_max = 1e-10;

    // initializing the vectors for calculating
    auto boundary = generate_container<double>(num_split,0);
    boundary[0] = boundary_val_l*hsq;
    boundary[num_split-1] = boundary_val_r*hsq;
    boundary[static_cast<int>(0.5*num_split)] = 2.0; //arbitrary value

    auto vals = generate_container<double>(num_split,0);
    vals[0] = boundary[0];
    vals[num_split-1] = boundary[num_split-1];

    auto change = generate_container<double>(num_split,0);
    change[0] = boundary[0];
    change[num_split-1] = boundary[num_split-1];

    auto deviation = generate_container<double>(num_split,0);

    //deviation to work with
    double d = 1.0;
    int count = 0;

    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel shared(vals,boundary,change,deviation) firstprivate(num_split) default(none)
    {
      for(int iter=0; iter<10000; ++iter){
#pragma omp for 
	for( int i=1 ; i < num_split-1; i++){
	    change[i] = func(vals[i-1],vals[i+1],boundary[i]);
	    deviation[i] = std::fabs(change[i]-vals[i]);
	}//end of parallel

#pragma omp for
	for( int i=1 ; i < num_split-1; i++){
	    vals[i] = change[i];
	}//end of parallel
	++count;
      }
    }
    auto end = std::chrono::high_resolution_clock::now();

    //output
    {
      std::string fn{"/home/s-nnschu/parallel_prog/values.txt"};
      std::ofstream out(fn);
      if(!out.good()){
	std::cout<<"Could not open file "<<fn<<std::endl;
      }else{
	for(int k=0; k<num_split;k++){
	    out << vals[k] << std::endl;
	}
      }
    }

    std::cout << "Elapsed time in seconds : " 
	      << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count()
	      << " sec"<< " " << count << std::endl;

    return 0;
}
