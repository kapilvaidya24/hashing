#include<iostream>
#include<algorithm>
#include<cmath>
#include <random>
#include <fstream>
// #include <quadmath.h>
#include "/home/kapilv/PhD_Acads/learned_hash/PGM-index/include/pgm/pgm_index.hpp"
#include <cstring>
using namespace std;



void get_slope_bias(vector<uint64_t> &v_int,int start,int end,double &slope,double &bias)
{
  double X=0.0,Y=0.0,XX=0.0,XY=0.0;

  for(int i=start;i<end;i++)
  {
    double x=v_int[i]-v_int[start];
    X+=x;
    Y+=i-start;
    XX+=x*x;
    XY+=x*(i-start);
  }
  double n_val=end-start;
  slope=(n_val*XY-X*Y)*1.00/(n_val*XX-X*X);
  bias=(Y-slope*X)*1.00/n_val;

  return ;
}



int main(int argc, char** argv){

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);

  // long long num_elements=2000000;

  // vector<double> v(num_elements,0.0);

  // for (int i=0; i<num_elements; i++) 
  // {
  //   v[i] = distribution(generator);
  // }

  
  cout<<"reading file "<<argv[1]<<endl;
  string file_name;
  string arg1(argv[1]);

  int c_check=0;
  if (arg1.compare("wiki")==0)
  {
    c_check=1;
    cout<<"wiki"<<endl;
    file_name= "data/wiki_ts_200M_uint64";  
  }

  if (arg1.compare("osm_cellids")==0)
  {
    c_check=1;
    cout<<"osm_cellids"<<endl;
     file_name="data/osm_cellids_200M_uint64";
  }

  if (arg1.compare("books64")==0)
  {
    c_check=1;
    cout<<"books64"<<endl;
    file_name="data/books_200M_uint64";
  }

  if (arg1.compare("fb")==0)
  {
    c_check=1;
    cout<<"fb"<<endl;
    file_name="data/fb_200M_uint64";
  }

  if(c_check==0)
  {
    if (arg1.compare("books32")==0)
    {
      cout<<"books32"<<endl;
      file_name="data/books_200M_uint32";
      
    }
  }

  std::ifstream input( file_name, std::ios::binary );  
  std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(input), {});
  
  cout<<"done reading file"<<endl;

  uint64_t file_elements=0;

  file_elements = buffer[0] | (buffer[1] << 8) | (buffer[2] << 16) | (buffer[3] << 24) | buffer[4+0] << 32 | (buffer[4+1] << 40) | (buffer[4+2] << 48) | (buffer[4+3] << 56);
  cout<<"value is: "<<file_elements<<" buffer size: "<<buffer.size()<<" c check: "<<c_check<<endl; 
  
  // file_elements=file_elements/2.00;  

  vector<uint64_t> v_int(file_elements,0.0);

  if(c_check==1)
  {
    for(uint64_t i=0;i<file_elements;i++)
    {
      uint64_t index=i*8+8;
      v_int[i]=buffer[index+0] | (buffer[index+1] << 8) | (buffer[index+2] << 16) | (buffer[index+3] << 24) | buffer[index+4+0] << 32 | (buffer[index+4+1] << 40) | (buffer[index+4+2] << 48) | (buffer[index+4+3] << 56);
    }
    cout<<"done building array "<<endl;
  }
  else
  {
    for(uint64_t i=0;i<file_elements;i++)
    {
      uint64_t index=i*4+8;
      v_int[i]=buffer[index+0] | (buffer[index+1] << 8) | (buffer[index+2] << 16) | (buffer[index+3] << 24);
    }

    cout<<"done building array "<<endl; 
  }

  std::sort(v_int.begin(), v_int.end());
  v_int.erase( unique( v_int.begin(), v_int.end() ), v_int.end() );

  file_elements=v_int.size();

  cout<<"done sorting de duplicating array "<<v_int.size()<<" max value is: "<<log2(v_int[v_int.size()-1])<<" max value is: "<<log2(v_int[v_int.size()-2])<<endl;

  int batch_size=100;
  long long num_loops=v_int.size()/batch_size;
  double slope,bias;
  long long cur_index;
  vector<int> bit_val(batch_size,0);
  double set_count=0;

  for(int i=0;i<num_loops;i++)
  {
    for(int j=0;j<batch_size;j++)
    {
      bit_val[j]=0;
    }

    cur_index=i*batch_size;

    get_slope_bias(v_int,cur_index,cur_index+batch_size,slope,bias);
    
    for(int j=cur_index;j<cur_index+batch_size;j++)
    {
      double index_doub=slope*(v_int[j]-v_int[cur_index])+bias;
      int index=index_doub;
      index=min(batch_size-1,index);
      index=max(0,index);
      bit_val[index]=1;
    }

    long long local_count=0;

    for(int j=0;j<batch_size;j++)
    {
      local_count+=bit_val[j];
    }
    set_count+=local_count;
    // cout<<i<<" batch empty spaces"<<batch_size-local_count<<" bound: "<<v_int[cur_index]<<" "<<v_int[cur_index+batch_size]<<endl;

  }

  double empty_count_our=file_elements-set_count;

  cout<<"proportion collisions: "<<empty_count_our*1.00/file_elements<<endl;

  // return 0;


  const int epsilon = 10; // space-time trade-off parameter
  pgm::PGMIndex<uint64_t, epsilon> index(v_int);

  cout<<"done building index "<<endl;

  vector<int> set_bit(file_elements,0);
  uint64_t zero=0;

  for(int i=0;i<file_elements-100;i++)
  {
    // if(i%10000==0)
    // {
    //   cout<<i<<" batch size"<<endl;
    // }
    auto range = index.search(v_int[i]);
    uint64_t val=range.pos;
    val=min(file_elements-1,val);
    val=max(zero,val);
    set_bit[range.pos]=1;
  }

  cout<<"done setting bits "<<endl;

  double count=0;

  for(int i=0;i<file_elements;i++)
  {
    count+=set_bit[i];
  }

  double empty_count=file_elements-count;

  cout<<"proportion collisions PGM: "<<empty_count*1.00/file_elements<<endl;

  cout<<"data size"<<v_int.size()<<" epsilon: "<<epsilon<<endl;

  cout<<"pgm stats: "<<index.segments_count()<<" height: "<<index.height()<<" size bytes: "<<index.size_in_bytes()<<endl;


  return 0;
}