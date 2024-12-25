


kk = 1;
for iii = 1:40
    for jjj = 1:50
        jj = jjj-10;
        ii = iii;
        if jjj <=40&&jjj>10%&&iii>10&&iii<90
            if ii >10&&ii<=24%&&mod(ii,2)==0
                pos(kk,:)=[Xc+ii*1-20, -15+jj*1, (ii-10)*1];
                kk = kk+1;
            elseif ii >24&&ii<36%&&mod(ii,2)==0
                pos(kk,:)=[Xc+ii*1-20, -15+jj*1, (24-10)*1];
                kk = kk+1;
            else
                pos(kk,:)=[Xc+ii*1-20, -15+jj*1, 0];
                kk = kk+1;
            end
        else
            pos(kk,:)=[Xc+iii*1-20, -25+jjj*1, 0];
            kk = kk+1;
        end
    end
end

x_real = pos(:,1);
y_real = pos(:,2);
z_real = pos(:,3);

X_calu = XX_new;
Y_calu = YY_new;
Z_calu = ZZ_new;

clear error_all
error_all = 0;
k=0;
for i = 1:length(X_calu)
    coord_x = (X_calu(i));
    coord_y = (Y_calu(i));
    coord_z = (Z_calu(i));
    coord = [coord_x,coord_y,coord_z];
    if coord_z>16||coord_z<-3
        continue
    end
    k=k+1;
    error_now1 = sum((pos - coord).^2,2);
    error_now =  sqrt(min(error_now1));
%     error_now1 = sum(abs(pos - coord),2);
%     error_now =  abs(min(error_now1));

    error_all = error_all + error_now;
      
end

mean = (error_all/k)