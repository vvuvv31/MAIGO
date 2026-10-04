#pragma once
#include "carbon/electron_multiple_cdf.hpp"
#include "carbon/electron_charged_data.hpp"
#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
namespace multiple_fixture {
namespace c=carbon::electron_condensed_research;namespace m=carbon::electron_multiple_cdf_research;
struct Law {unsigned node;float energy;c::SoftLaw soft;m::View table;};
struct Data {
 sycl::queue queue;Law* laws{};float* elastic{};float* quantiles{};unsigned count{},max_count{},bins{},knots{};
 Data(sycl::queue q,const std::filesystem::path& charged,const std::string& charged_sha,const std::filesystem::path& meta,const std::string& meta_sha):queue(q){
  if(carbon::compute_file_sha256_hex(charged)!=charged_sha||carbon::compute_file_sha256_hex(meta)!=meta_sha)throw std::runtime_error("fixture source pin");
  std::ifstream in(meta);std::string magic,source,file,pin;in>>magic>>source>>count>>max_count>>bins>>knots>>file>>pin;
  if(!in||magic!="MAIMULT1"||source!=charged_sha||!count||max_count<2||max_count>128||bins!=32||knots!=48*bins+1)throw std::runtime_error("fixture metadata");
  const auto path=meta.parent_path()/file;const auto values=std::size_t(count)*(max_count-1)*knots;
  if(carbon::compute_file_sha256_hex(path)!=pin||std::filesystem::file_size(path)!=values*sizeof(float))throw std::runtime_error("fixture table pin/size");
  laws=sycl::malloc_shared<Law>(count,queue);quantiles=sycl::malloc_shared<float>(values,queue);elastic=sycl::malloc_shared<float>(82*31*38,queue);
  if(!laws||!quantiles||!elastic)throw std::bad_alloc();
  std::ifstream raw(path,std::ios::binary);raw.read(reinterpret_cast<char*>(quantiles),values*sizeof(float));if(!raw)throw std::runtime_error("fixture read");
  for(unsigned i=0;i<count;++i){auto& l=laws[i];in>>l.node>>l.energy>>l.soft.screening>>l.soft.cutoff_q>>l.soft.lower_row_probability>>l.soft.mean_q>>l.soft.second_q>>l.soft.lower_row;
   if(!in||l.node>=82||l.soft.lower_row>=30||!(l.soft.screening>0)||!(l.soft.cutoff_q>0)||l.soft.cutoff_q>.01f||!(l.soft.lower_row_probability>=0&&l.soft.lower_row_probability<=1))throw std::runtime_error("fixture law");
   l.table={quantiles+std::size_t(i)*(max_count-1)*knots,max_count,bins};
   for(unsigned n=2;n<=max_count;++n){const float* row=l.table.quantiles+(n-2)*knots;if(row[0]!=0||row[knots-1]>2)throw std::runtime_error("fixture support");for(unsigned k=0;k<knots;++k)if(!std::isfinite(row[k])||(k&&row[k]<row[k-1]))throw std::runtime_error("fixture monotonicity");}
  }
  std::string extra;if(in>>extra)throw std::runtime_error("fixture trailing data");
  std::ifstream index(charged);unsigned nodes;index>>magic>>nodes;if(magic!="MAIECB01"||nodes!=82)throw std::runtime_error("charged metadata");
  for(unsigned i=0;i<nodes;++i){int section;double density;index>>section>>density>>file>>pin;if(!index)throw std::runtime_error("charged index");
   bool used=false;for(unsigned j=0;j<count;++j)used=used||laws[j].node==i;if(!used)continue;
   carbon::ChargedResponseData data(charged.parent_path()/file,pin);if(section!=data.section||density!=data.density)throw std::runtime_error("charged identity");
   for(unsigned k=0;k<31*38;++k)elastic[i*31*38+k]=float(data.elastic_rows[k]);
   for(unsigned j=0;j<count;++j)if(laws[j].node==i)laws[j].soft.elastic={elastic+i*31*38,float(data.mass),float(data.bc),float(data.xc2)};
  }
 }
 ~Data(){sycl::free(laws,queue);sycl::free(quantiles,queue);sycl::free(elastic,queue);}
};
inline std::uint64_t seed(std::uint64_t value){value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;value=(value^(value>>27))*0x94d049bb133111ebULL;return value^(value>>31);}
}
