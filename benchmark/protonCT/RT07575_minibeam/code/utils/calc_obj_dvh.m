function [obj,d]=calc_obj_dvh(x,var)
Dij=var.Dij;
N_obj=var.N_obj;
id_obj=var.id_obj;
w_obj=var.w_obj;
c_obj=var.c_obj;
s_obj=var.s_obj;
N_c=var.N_c;
n_c=var.n_c;
c=var.c;
N_dij=var.N_dij;

obj=zeros(N_obj,N_dij);
d=cell(N_c,N_dij);

for j=1:N_dij
y1=Dij{j}*double(x);

for i=1:N_obj
    if ~isempty(id_obj{i,j})
    obj(i,j)=w_obj(i,j)/n_c(c_obj(i))*sum((y1(id_obj{i,j})-s_obj(i)).^2);
    end
end


for i=1:N_c
    d{i,j}=y1(c{i});
end
end