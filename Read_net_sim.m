[mm,nn] = size(X_result);
X_figure = X_result;
% X_figure = X_result(:,1:nn/2) + 1j * X_result(:,nn/2+1:end);
% 
delta = 0.2;
s = [-5:delta:12];
Hes_all = zeros( size(X,1), size(X,2),size(s,2)) + nan;
Hes = zeros(size(X,1), size(X,2)) + nan;
% s = -25:1:60;
for i = 1:mm
%     curve_MAP = abs( X_figure(i,:))/max(abs( X_figure(i,:)));
%     curve_extremum = intersect(find(diff(curve_MAP)>0)+1,find(diff(curve_MAP)<0));
%     image_3D(curve_extremum,index_azimuth,index_range) = hat(curve_extremum);

    Hes_all(index_x_choose(i), index_y_choose(i),:) = X_figure(i,:);
    index = find( abs(X_figure(i,:)) == max(abs(X_figure(i, :))) );

    if length(index) <5&& isempty(index)==0
        Hes(index_x_choose(i), index_y_choose(i)) = s_all(i,index);
    end
%     if TarCol(i) == 1010 && TarRow(i) == 820
%         aaa = 1;
%     end
end
% curve_MAP = abs(hat)/max(abs(hat));
% curve_extremum = intersect(find(diff(curve_MAP)>0)+1,find(diff(curve_MAP)<0));
% image_3D(curve_extremum,index_azimuth,index_range) = hat(curve_extremum);

figure();
imagesc(Hes)
colormap('jet')

% figure();imagesc(squeeze(abs(Hes_all(:,256,:))).')
% 
% figure();plot(squeeze(abs(Hes_all(377,747,:))));
% 
% figure();
% imagesc(squeeze(abs(Hes_all(140,50:150,:))))
% figure();
% plot(squeeze(abs(Hes_all(140,95,:))))
% figure();
% imagesc(squeeze(abs(image_3D(:,140,50:150))).')
%% least square


%% point abstract
clear XX YY ZZ ZZ_h
ii = 1;
for iii = 1:length(index_x_choose)
    i = index_x_choose(iii);
    j = index_y_choose(iii);
    maxh = max(squeeze(abs(Hes_all(i,j,:) )));
    for k = 1:size(image_3D,1)
        if(abs(Hes_all(i,j,k)) >0.5*maxh)% && X(i,j)<804)
%             if s(k) == 0 || s(k) == 5
                XX(ii) = X(i,j);
                YY(ii) = Y(i,j);
                ZZ(ii) = s_all(iii,k);
                ZZ_h(ii) = abs(Hes_all(i,j,k));
                ii = ii+1;
%             end
        end
    end
end

figure()
scatter3(XX(1:1:end),YY(1:1:end),ZZ(1:1:end),10,ZZ_h(1:1:end))
colormap('jet')
axis([Xc-15 Xc+15 -20 20 -5 40])
view([-40 -45 15])

%% coordinate transform
clear XX_new YY_new ZZ_new ZZ_h_new
for i = 1:ii-1

    x = XX(i);
    y = YY(i);
    z = ZZ(i);

    RadarPos_now = RadarPos(MasterNum,:);
    R2 = (RadarPos_now(1)-x)^2 + (RadarPos_now(2)-y)^2 + RadarPos_now(3)^2;
    
    theta = asind(sqrt((RadarPos_now(1)-x)^2 + (RadarPos_now(2)-y)^2)/sqrt(R2));
    fa = atand(y/x);
    delta_z = z*sind(theta);
    delta_x = z*cosd(theta)*cosd(fa);
    delta_y = z*cosd(theta)*sind(fa);

    XX_new(i) = x + delta_x;
    YY_new(i) = y + delta_y;
    ZZ_new(i) = (delta_z);
    ZZ_h_new(i) = ZZ_h(i);
%     if ZZ_h_new(i)>1500
%         ZZ_h_new(i) = 1500;
%     end
end

figure()
scatter3(XX_new(1:1:end),YY_new(1:1:end),ZZ_new(1:1:end),25,ZZ_new(1:1:end))
% scatter3(1,1,1,10,1,'filled')
colormap('jet')
axis([Xc-30 Xc+30 -30 30 -3 16])
caxis([1 15])
% axis([Xc-15 Xc+15 -40 0 -5 30])
view([0 -90 0])
view([-40 -60 45])%15




















% clear all;
% close all;

%% net output read
[mm,nn] = size(X_result);
X_figure = X_result;
Hes_all = zeros( size(X,1), size(X,2),size(s,2)) + nan;
Hes = zeros(size(X,1), size(X,2)) + nan;
% s = -25:1:60;
for i = 1:mm
    Hes_all(index_x_choose(i), index_y_choose(i),:) = X_figure(i,:);
    index = find( abs(X_figure(i,:)) == max(abs(X_figure(i, :))) );

    if length(index) <5&& isempty(index)==0
        Hes(index_x_choose(i), index_y_choose(i)) = s_all(i,index);
    end

end

figure();
imagesc(Hes)
colormap('jet')


%% point abstract
clear XX YY ZZ ZZ_h
ii = 1;
for iii = 1:length(index_x_choose)
    i = index_x_choose(iii);
    j = index_y_choose(iii);
    maxh = max(squeeze(abs(Hes_all(i,j,:) )));
    for k = 1:size(image_3D,1)
        if(abs(Hes_all(i,j,k)) >0.75*maxh)

                XX(ii) = X(i,j);
                YY(ii) = Y(i,j);
                ZZ(ii) = s_all(iii,k);
                ZZ_h(ii) = abs(Hes_all(i,j,k));
                ii = ii+1;
%             end
        end
    end
end

%% coordinate transform
clear XX_new YY_new ZZ_new ZZ_h_new
for i = 1:ii-1

    x = XX(i);
    y = YY(i);
    z = ZZ(i);

    RadarPos_now = RadarPos(MasterNum,:);
    R2 = (RadarPos_now(1)-x)^2 + (RadarPos_now(2)-y)^2 + RadarPos_now(3)^2;
    
    theta = asind(sqrt((RadarPos_now(1)-x)^2 + (RadarPos_now(2)-y)^2)/sqrt(R2));
    fa = atand(y/x);
    delta_z = z*sind(theta);
    delta_x = z*cosd(theta)*cosd(fa);
    delta_y = z*cosd(theta)*sind(fa);

    XX_new(i) = x + delta_x;
    YY_new(i) = y + delta_y;
    ZZ_new(i) = (delta_z);
    ZZ_h_new(i) = ZZ_h(i);
end

figure()
scatter3(XX_new(1:1:end),YY_new(1:1:end),ZZ_new(1:1:end),25,ZZ_new(1:1:end))
colormap('jet')
axis([Xc-30 Xc+30 -30 30 -3 16])
caxis([1 15])
view([0 -90 0])
view([-40 -60 45])%15
set(gcf,'color','white'); %窗口背景黑色
colordef white; %2D/3D图背景黑色
