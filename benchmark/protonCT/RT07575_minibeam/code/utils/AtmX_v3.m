function x=AtmX_v3(y,var)
Dij=var.Dij;
id_obj=var.id_obj;
N_obj=var.N_obj;
w_obj=var.w_obj;
n_c=var.n_c;
c_obj=var.c_obj;
nX=var.nX;
nY=var.nY;
N_dij=var.N_dij;
x=zeros([nX,1]);
for j=1:N_dij
    y0=zeros([nY,1]);
    for i=1:N_obj
        id=id_obj{i,j};
        if ~isempty(id)
            y0(id)=y0(id)+w_obj(i,j)/n_c(c_obj(i))*double(y{i,j});
        end
    end
    x=x+Dij{j}'*y0;
end
x=single(x);
end