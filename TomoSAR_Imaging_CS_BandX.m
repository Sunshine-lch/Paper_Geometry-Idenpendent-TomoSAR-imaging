close all;
clear all;

path = './SimData_BandX';

load([path, '/sence.mat'])
%% sim para
c = c;
fc = fc;
lambda = c/fc;

% %% Radar_elevation_pos
radarX =0*ones(H_num,1);
radarY = 0*zeros(H_num,1);
RadarPos = [radarX radarY Hort_z];
MasterNum = 11;
base_pos = Hort_z;


%% torget_pos
load([path,'/f_back_1.mat'])
g_all = zeros([H_num,size(f_back,1),size(f_back,2)]) + j*zeros([H_num,size(f_back,1),size(f_back,2)]);
for i = 1:H_num
    load([path,'/f_back_',num2str(i),'.mat'])
    g_all(i,:,:) = f_back; 
end

image_2D =  squeeze(sum(abs(g_all)));
aa = max(max(abs(image_2D)));
bb = abs(image_2D);
cc = bb/aa;
[index_x_choose, index_y_choose] = find(cc > 0.3);
dd = nan*ones(size(image_2D));
dd(index_x_choose, index_y_choose) = cc(index_x_choose, index_y_choose);

image_2D_show = nan*ones(size(image_2D));
image_2D_show(index_x_choose, index_y_choose) = image_2D(index_x_choose, index_y_choose);
figure();imagesc(abs(image_2D_show))

sence.x = X;
sence.y = Y;
delta = 0.2;
s = [-5:delta:12];
image_3D = zeros([length(s),size(g_all,2),size(g_all,3)]) + nan;

%% observation model obtained

L_all = zeros(length(index_x_choose),H_num,length(s));
g_re_all = zeros(length(index_x_choose),H_num);
s_all = zeros(length(index_x_choose),length(s));
for TarTomoNum = 1:length(index_x_choose)
    if mod(TarTomoNum,100) == 0
        disp(['Bar：' num2str(TarTomoNum/length(index_x_choose)*100)])
    end

    index_azimuth = index_x_choose(TarTomoNum);
    index_range = index_y_choose(TarTomoNum);
    pos_X = sence.x(index_azimuth,index_range);
    pos_Y = sence.y(index_azimuth,index_range);

    TargetPos = [pos_X pos_Y 0 ];

    Los_vec = TargetPos - RadarPos(MasterNum,:);

    V_vec = [0,1,0];
    S_vec = cross( Los_vec,V_vec ); 
    S_vec = S_vec / norm(S_vec);

    N = H_num;
    Bn = zeros(N,3);
    Bs = zeros(N,1);

    Rang_Rad2Tar = zeros(N,1);
    for ii = 1:N

        Bn(ii,:) = RadarPos(ii,:) - RadarPos(MasterNum,:);
        Bs(ii) = dot( Bn(ii,:),S_vec );

        Rang_Rad2Tar(ii) = norm( [RadarPos(ii,1)-pos_X  RadarPos(ii,2)-pos_Y  RadarPos(ii,3)-0] );
    end
    H_sort = Bs;%temp_H;
    Range_master = Rang_Rad2Tar(MasterNum);

    
    H_sort1 = sort(H_sort);
    DB = H_sort1(end)-H_sort1(1);
    base_pos = H_sort;
    rhos = lambda*Range_master/2/DB;

    s = (-5:0.2:12)*rhos;
    xi = (-2*base_pos/lambda/Range_master);
    L = exp(-1j*2*pi*xi*s)/sqrt(H_num);

    R0 = Rang_Rad2Tar;
    Echo_Ref = exp(1j*4*pi*R0/lambda);
    echo = squeeze(g_all(:,index_azimuth,index_range));
    g = echo .* Echo_Ref;

    L_all(TarTomoNum,:,:) = L;
    g_re_all(TarTomoNum,:) = g;
    s_all(TarTomoNum,:) = s;

end

tic
for ii = 1:length(index_x_choose)
    if mod(ii,200) == 0
        disp(ii/length(index_x_choose))
    end
    
    index_azimuth = index_x_choose(ii);
    index_range = index_y_choose(ii);

    L = L_all(ii,:,:);
    L = squeeze(L);
    g = g_re_all(ii,:);
    g = squeeze(g).';


%% ISTA
    yhat = [real(g).' imag(g).'];
    Ahat = [real(L) -imag(L)
        imag(L) real(L)];
    yhat = yhat./max(abs(yhat));
    

    N2 = size(L,2);
    [theta, record_e] = lasso_ista(yhat.',Ahat,1,1000);%
    
    hatreal = theta(1:N2);
    hatimag = theta(N2+1:end);
    hat = hatreal + 1j*hatimag;

    curve_MAP = abs(hat)/max(abs(hat));
    image_3D(:,index_azimuth,index_range) = curve_MAP;


end

clear XX YY ZZ ZZ_h
ii = 1;
for iii = 1:length(index_x_choose)
    i = index_x_choose(iii);
    j = index_y_choose(iii);
    maxh = max(squeeze(abs(image_3D(:,i,j) )));
    for k = 1:size(image_3D,1)
        if(abs(image_3D(k,i,j)) > 0.5*maxh) %0.5
%             if s(k) == 0 || s(k) == 5
                XX(ii) = X(i,j);
                YY(ii) = Y(i,j);
                ZZ(ii) = s_all(iii,k);
                ZZ_h(ii) = abs(image_3D(k,i,j));
                ii = ii+1;
%             end
        end
    end
end

%%%%%%%%%%%%% Adgust
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

