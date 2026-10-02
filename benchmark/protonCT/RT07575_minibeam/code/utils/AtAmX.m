
function Y=AtAmX(X,AmX,AtmX,var)
Y=AtmX(AmX(X,var),var);
% if var.isC==1&&isreal(Y)
%     Y=complex(Y,zeros(size(Y),'single'));
% end
