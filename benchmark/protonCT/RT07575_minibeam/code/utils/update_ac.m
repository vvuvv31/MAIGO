function [rhs,var]=update_ac(x,var)
Dij=var.Dij;
c=var.c;
N_obj=var.N_obj;
n_obj=var.n_obj;
s_obj=var.s_obj;
c_obj=var.c_obj;
type_obj=var.type_obj;
N_dij=var.N_dij;
rhs=cell(N_obj,N_dij);
id_obj=cell(N_obj,N_dij);

for j=1:N_dij
y1=Dij{j}*double(x);

for i=1:N_obj
    if type_obj(i)==1 % max constraint
        id=c{c_obj(i)};
        if isempty(id), continue; end
        y=y1(id);
        [y2,I]=sort(y,'descend');
        id=id(I);
        if y2(n_obj(i))>s_obj(i)
            id1=find(y2<=1.01*y2(n_obj(i)),1,'first');
            id2=find(y2<=0.99*s_obj(i),1,'first');
            if isempty(id1), id1=n_obj(i); end
            if isempty(id2), id2=numel(id); end
            if id2<id1, continue; end
            id_obj{i,j}=id(id1:id2);                
            rhs{i,j}=ones([numel(id_obj{i,j}) 1])*s_obj(i);
        end
    elseif type_obj(i)==2 % min constraint
        id=c{c_obj(i)};
        if isempty(id), continue; end
        y=y1(id);
        [y2,I]=sort(y,'descend');
        id=id(I);
        if y2(n_obj(i))<s_obj(i)
            id1=find(y2<s_obj(i),1,'first');
            id2=find(y2<=0.99*y2(n_obj(i)),1,'first');
            if isempty(id1), id1=n_obj(i); end
            if isempty(id2), id2=numel(id); end
            if id2<id1, continue; end
            id_obj{i,j}=id(id1:id2);
            rhs{i,j}=ones([numel(id_obj{i,j}) 1])*s_obj(i);
        end
    elseif type_obj(i)==3 % max ptv/body
        id=c{c_obj(i)};
        y=y1(id);
        id2=find(y>s_obj(i));
        if ~isempty(id2)
            id_obj{i,j}=id(id2);
            rhs{i,j}=ones([numel(id_obj{i,j}) 1])*s_obj(i);
        end
    elseif type_obj(i)==0
        id_obj{i,j}=c{c_obj(i)};
        rhs{i,j}=ones([numel(id_obj{i,j}) 1])*s_obj(i);
    end
end
end
var.id_obj=id_obj;
end
