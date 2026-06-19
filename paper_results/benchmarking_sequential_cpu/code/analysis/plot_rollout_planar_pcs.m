clear all;
close all;
clc;

script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir, '..', '..', 'data');

FontLabels      = 32;
% axes1 = axes('FontSize',34);box(axes1,'on');hold(axes1,'on');
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', FontLabels);
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

colors = [0 0.6 1;  % blue
    0.55 0.27 0.07;
    0.4660 0.6740 0.1880;  % green
    1 0.1 0.1];


%%
kk=1;

load(fullfile(data_dir, 'end_effector_position_planar_pcs_sorosim.mat'))
plot(t(1:kk:end),Pos_EE_all(1,(1:kk:end)),'LineWidth',3,'Color', colors(1,:));hold on;grid on;
plot(t(1:kk:end),Pos_EE_all(2,(1:kk:end)),'LineWidth',3,'Color', colors(2,:));hold on;grid on;
plot(t(1:kk:end),Pos_EE_all(3,(1:kk:end)),'LineWidth',3,'Color', colors(3,:));hold on;grid on;
xlabel('Time $ (\mathrm{s})$','Interpreter','latex','FontName','Times New Roman','FontSize',FontLabels);
ylabel('Tip Coordinates $(\mathrm{m})$','Interpreter','latex','FontName','Times New Roman','FontSize',FontLabels);
% set(gca, 'XTickLabel', []);

% kk=20;

T = readtable(fullfile(data_dir, 'end_effector_position_planar_pyelastica.xlsx'));
kkk=15;
hold on;plot(T.time(1:kk:end),T.x(1:kk:end),'LineWidth',4,'Color', colors(1,:),'LineStyle','--');hold on;
hold on;plot(T.time(1:kk:end),T.y(1:kk:end),'LineWidth',4,'Color', colors(2,:),'LineStyle','--');hold on;
% hold on;plot(T1.Var1(1:kkk:end),T1.Var2(1:kkk:end),'LineWidth',2,'Color', colors(1,:),'LineStyle','none','Marker','square');hold on;
plot(T.time(1:kk:end),T.z(1:kk:end),'LineWidth',4,'Color', colors(3,:),'LineStyle','--');


T1 = readtable(fullfile(data_dir, 'end_effector_position_planar_pcs_soromox.csv'));
kkk=10;

hold on;plot(T1.t(1:kk:end),T1.x(1:kk:end),'LineWidth',4,'Color', colors(1,:),'LineStyle','-.');hold on;
% hold on;plot(T1.Var1(1:kkk:end),T1.Var2(1:kkk:end),'LineWidth',2,'Color', colors(1,:),'LineStyle','none','Marker','o');hold on;
plot(T1.t(1:kk:end),T1.x(1:kk:end)*0,'LineWidth',4,'Color', colors(2,:),'LineStyle','-.');hold on;
plot(T1.t(1:kk:end),T1.z(1:kk:end),'LineWidth',4,'Color', colors(3,:),'LineStyle','-.');

legend({'$x_t$ (SoRoSim)', '$y_t$ (SoRoSim)','$z_t$ (SoRoSim)', ...
                '$x_t$ (PyElastica)','$y_t$ (PyElastica)','$z_t$ (PyElastica)'...
                '$x_t$ (SoRoMoX)','$y_t$ (SoRoMoX)','$z_t$ (SoRoMoX)', ...
                '$x_t$ (PyElastica)','$y_t$ (PyElastica)','$z_t$ (PyElastica)'}, ...
        'Interpreter','latex', ...
        'Location','north', ...
        'Orientation','horizontal', ...
        'NumColumns',3);

