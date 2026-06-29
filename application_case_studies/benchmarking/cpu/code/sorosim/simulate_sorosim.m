startup

clear all;
close all;
clc;

script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir, '..', '..', 'data');

FontLabels      = 32;
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', FontLabels);
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');

colors = [0 0.6 1;  % blue
    0.55 0.27 0.07;
    0.4660 0.6740 0.1880;  % green
    1 0.1 0.1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% I RUN THE SELECTED CASE: Case = 1: Planar PCS, Case = 2: Spatial PCS,
% Case = 3: Complex GVS, Case 4: Tendon-driven GVS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Case = 4;

switch Case
   case 1
      load 'system_planar_pcs.mat';
      output_file = 'end_effector_position_planar_pcs_sorosim.mat';
   case 2
      load 'system_spatial_pcs.mat';
      output_file = 'end_effector_position_spatial_pcs_sorosim.mat';
   case 3
      load 'system_complex_gvs.mat';
      output_file = 'end_effector_position_complex_gvs_sorosim.mat';
   otherwise
      load 'system_tendon_driven_gvs.mat';
      output_file = 'end_effector_position_tendon_driven_gvs_sorosim.mat';
end

S1.PlotParameters.CameraPosition = [0 -6.235382907247957 0];
S1.PlotParameters.CameraTarget = [0 0 0];
S1.PlotParameters.CameraUpVector = [0 0 1];

% Simulation seettings

tmax                    = 3;
h                       = 0.0001;
odetype                 = 'ode45'; 
global gv;
gv.tmax                 = tmax;
qqd0      = zeros(2*S1.ndof,1);

tic

[t,qqd] = dynamics(S1,qqd0,odetype,h);

% S1.plotqqd(t,qqd)
time_comp = toc;


fprintf("Computation time: %.6f s\n", time_comp);

%%
Pos_EE_all  = [];
for i=1:length(t)    
    q_here                  = qqd(i,1:end/2);
    g_all                   = FwdKinematics(S1,q_here);
    g_EE                    = g_all(end-3:end,end-3:end);
    Pos_EE                  = g_EE(1:3,4);
    Pos_EE_all              = [Pos_EE_all Pos_EE];
end

if ~exist(data_dir, 'dir')
    mkdir(data_dir);
end
save(fullfile(data_dir, output_file), 't', 'Pos_EE_all');

%%

plot(t,Pos_EE_all(1,:),'LineWidth',3,'Color', colors(1,:));hold on;grid on;
plot(t,Pos_EE_all(2,:),'LineWidth',3,'Color', colors(2,:));hold on;grid on;
plot(t,Pos_EE_all(3,:),'LineWidth',3,'Color', colors(3,:));hold on;grid on;
xlabel('Time $(\mathrm{s})$','Interpreter','latex','FontName','Times New Roman','FontSize',FontLabels);
ylabel('Tip Coordinates $(\mathrm{m})$','Interpreter','latex','FontName','Times New Roman','FontSize',FontLabels);

legend({'$x_t$ (SoRoSim)', '$y_t$ (SoRoSim)','$z_t$ (SoRoSim)'}, ...
        'Interpreter','latex', ...
        'Location','north', ...
        'Orientation','horizontal', ...
        'NumColumns',3);






