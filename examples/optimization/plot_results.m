clear all %#ok<CLALL>
close all
clc


%% Load data
fname = 'optimization_results.mat';

% Collocated
dir_c = 'data\collocated\trial3';
data_c = load(fullfile(dir_c, fname));

% Synergistic
dir_s = 'data\synergistic\trial6';
data_s = load(fullfile(dir_s, fname));

% Save flag
savefigs = false;


%% Main figure
fig = figure(Color='w', Units='centimeters', Position=[0.5, 0.5, 28, 16]);
t = tiledlayout(fig, 2, 2, Padding='tight', TileSpacing='tight');

% Fontsize
fs = 0.4; % [cm]

% Colors for plots
linecolors = orderedcolors("gem");
linecolors([1, 3], :) = linecolors([3, 1], :);
fillcolors = orderedcolors("glow");
fillcolors([1, 3], :) = fillcolors([3, 1], :);


%% Loss
% Collocated
ydata = data_c.history_loss / 100;

% Calculate Statistics
xdata = 1:size(ydata, 1);
min_y = min(ydata, [], 2);
max_y = max(ydata, [], 2);

% Shaded Area (Min/Max)
x_poly = [xdata, fliplr(xdata)];
y_poly = [min_y', fliplr(max_y')];

% Plot
ax = nexttile(t, 1);
hold on; grid on; box on;

% Light blue color (RGB: [0.6 0.8 1])
fill(x_poly, y_poly, [0.6 0.8 1], ...
    'FaceAlpha', 0.5, ...
    'EdgeColor', 'none', ...
    'DisplayName', 'Min-Max Range');

% Min Line
plot(xdata, min_y, '-ob', LineWidth=1.5, MarkerSize=2, ...
    MarkerFaceColor='b', DisplayName='Best');

% Formatting
xlabel('Iterations', Interpreter='latex', FontSize=fs, Units="centimeters");
ylabel("Loss", Interpreter='latex', FontSize=fs, Units="centimeters");
xlim([1, max(xdata)]);
legend(Location="northeast", Interpreter='latex')
text(ax, 0.02, 0.95, '\textbf{(a)}', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');
fontsize(fs, "centimeters")


% Synergistic
ydata = data_s.history_loss;

% Calculate Statistics
xdata = 1:size(ydata, 1);
min_y = min(ydata, [], 2);
max_y = max(ydata, [], 2);

% Shaded Area (Min/Max)
x_poly = [xdata, fliplr(xdata)];
y_poly = [min_y', fliplr(max_y')];

% Plot
ax = nexttile(t, 3);
hold on; grid on; box on;

% Light blue color (RGB: [0.6 0.8 1])
fill(x_poly, y_poly, [0.6 0.8 1], ...
    'FaceAlpha', 0.5, ...
    'EdgeColor', 'none', ...
    'DisplayName', 'Min-Max Range');

% Min Line
plot(xdata, min_y, '-ob', LineWidth=1.5, MarkerSize=2, ...
    MarkerFaceColor='b', DisplayName='Best');

% Formatting
xlabel('Iterations', Interpreter='latex', FontSize=fs, Units="centimeters");
ylabel("Loss", Interpreter='latex', FontSize=fs, Units="centimeters");
xlim([1, max(xdata)]);
text(ax, 0.02, 0.95, '\textbf{(c)}', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');
fontsize(fs, "centimeters")


%% End-effector position
% Synergistic
ts = data_s.t_ts(1,:);

% Index of best simulation
[min_val_per_batch, ~] = min(data_c.history_loss);
[~, best_batch] = min(min_val_per_batch);

% 1. Load and standardise dimensions
raw_init = data_s.x_ts_init; % (Batch x Time x 3)
raw_best = data_s.x_ts_best; % (Batch x Time x 3)

% 2. Calculate Mean, Min, Max for Initial Batch
% init.mean = squeeze(mean(raw_init, 1));
% init.mean = squeeze(raw_init(best_batch,:,:));
init.mean = squeeze(median(raw_init, 1));
init.min  = squeeze(min(raw_init, [], 1));
init.max  = squeeze(max(raw_init, [], 1));

% 3. Calculate Mean, Min, Max for Best Batch
% best.mean = squeeze(mean(raw_best, 1));
best.mean = squeeze(raw_best(best_batch,:,:));
% best.mean = squeeze(median(raw_best, 1));
best.min  = squeeze(min(raw_best, [], 1));
best.max  = squeeze(max(raw_best, [], 1));

% Ensure time vector is a row vector for concatenation logic [t, fliplr(t)]
ts = reshape(ts, 1, []);

% Desired configuration
x_des_ts = data_s.x_des_ts;

ax = nexttile(t, 4);
hold on; grid on; box on;
for j=1:3
    lcol = linecolors(j,:);
    fcol = fillcolors(j,:);
    
    % --- A. Plot Target (Dashed) ---
    yline(x_des_ts(:, j), LineStyle='--', Color=lcol, LineWidth=2)
    
    % --- B. Plot Initial Batch (Area + Mean) ---
    % Filled Area (Min to Max)
    x_poly = [ts, fliplr(ts)];
    y_poly_init = [init.max(:, j)', fliplr(init.min(:, j)')];
    
    % Plot area with transparency (FaceAlpha) and no edges
    fill(x_poly, y_poly_init, fcol, 'FaceAlpha', 0.35, ...
        'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % Plot Mean Line
    plot(ts, init.mean(:, j), LineStyle="-.", LineWidth=1.5, Color=lcol);
    
    % --- C. Plot Optimized Batch (Area + Mean) ---
    y_poly_best = [best.max(:, j)', fliplr(best.min(:, j)')];
    
    % Plot area (slightly darker or same alpha)
    fill(x_poly, y_poly_best, fcol, 'FaceAlpha', 0.5, ...
        'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % Plot Mean Line
    plot(ts, best.mean(:, j), LineStyle="-", LineWidth=1.25, Color=lcol);
end
hold off
% Formatting
xlabel('Time [s]', Interpreter='latex', FontSize=fs, Units="centimeters");
ylabel('End-effector position [m]', Interpreter='latex', FontSize=fs, Units="centimeters");
xlim([0, 5]);
xticks(0:1:5);
xticklabels(["0", "1", "2", "3", "4", "5"]);
text(ax, 0.02, 0.95, '\textbf{(d)}', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'latex');


%% Configuration
% Collocated
ts = data_c.t_ts(1,:);

% Index of best simulation
[min_val_per_batch, ~] = min(data_c.history_loss);
[~, best_batch] = min(min_val_per_batch);

% 1. Load and standardise dimensions
raw_init = data_c.q_ts_init; % (Batch x Time x 6)
raw_best = data_c.q_ts_best; % (Batch x Time x 6)

% 2. Calculate Mean, Min, Max for Initial Batch
% init.mean = squeeze(mean(raw_init, 1));
% init.mean = squeeze(raw_init(best_batch,:,:));
init.mean = squeeze(median(raw_init, 1));
init.min  = squeeze(min(raw_init, [], 1));
init.max  = squeeze(max(raw_init, [], 1));

% 3. Calculate Mean, Min, Max for Best Batch
% best.mean = squeeze(mean(raw_best, 1));
best.mean = squeeze(raw_best(best_batch,:,:));
% best.mean = squeeze(median(raw_best, 1));
best.min  = squeeze(min(raw_best, [], 1));
best.max  = squeeze(max(raw_best, [], 1));

% 4. Get dimensions from the averaged data
[n_timesteps, n_vars] = size(init.mean);
n_seg = n_vars / 6;

% Ensure time vector is a row vector for concatenation logic [t, fliplr(t)]
ts = reshape(ts, 1, []);

% Desired configuration
q_des_ts = data_c.q_des_ts;

% Plot
i = 0;

% =====================================================================
% 1. Angular Strains Plot
% =====================================================================
ax = nexttile(t, 2);
hold on; grid on; box on;
for j = 1:3
    idx = 6*i + j; % Current variable index
    lcol = linecolors(j,:);
    fcol = fillcolors(j,:);
    
    % --- A. Plot Target (Dashed) ---
    plot(ts, q_des_ts(:, idx), LineStyle="--", LineWidth=2, Color=lcol);
    
    % --- B. Plot Initial Batch (Area + Mean) ---
    % Filled Area (Min to Max)
    x_poly = [ts, fliplr(ts)];
    y_poly_init = [init.max(:, idx)', fliplr(init.min(:, idx)')];
    
    % Plot area with transparency (FaceAlpha) and no edges
    fill(x_poly, y_poly_init, fcol, 'FaceAlpha', 0.35, ...
        'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % Plot Mean Line
    plot(ts, init.mean(:, idx), LineStyle="-.", LineWidth=1.5, Color=lcol);
    
    % --- C. Plot Optimized Batch (Area + Mean) ---
    y_poly_best = [best.max(:, idx)', fliplr(best.min(:, idx)')];
    
    % Plot area (slightly darker or same alpha)
    fill(x_poly, y_poly_best, fcol, 'FaceAlpha', 0.5, ...
        'EdgeColor', 'none', 'HandleVisibility', 'off');
    
    % Plot Mean Line
    plot(ts, best.mean(:, idx), LineStyle="-", LineWidth=1.25, Color=lcol);
end
h1 = plot(nan, nan, 'k--', LineWidth=2);
h2 = plot(nan, nan, 'k-.', LineWidth=1.5);
h3 = plot(nan, nan, 'k-', LineWidth=1.25);
hold off
% Formatting
legend([h1, h2, h3], {"Target", "Initial (median $\pm$ range)", "Optimized (best $\pm$ range)"}, ...
    Location="best", AutoUpdate="off", Interpreter='latex')
xlim([min(ts), max(ts)])
xlabel("Time [s]", Interpreter='latex', FontSize=fs, Units="centimeters")
ylabel("Angular strains [rad/m]", Interpreter='latex', FontSize=fs, Units="centimeters")
xlim([0, 5]);
xticks(0:1:5);
xticklabels(["0", "1", "2", "3", "4", "5"]);
text(ax, 0.02, 0.95, '\textbf{(b)}', 'Units', 'normalized', 'FontWeight', 'bold', 'Interpreter', 'latex');
fontsize(fs, "centimeters")


%% Save figure
if savefigs
    fname = strcat("plots.pdf");
    set(gcf,'Units','centimeters');
    screenposition = get(gcf, 'Position');
    set(gcf, PaperPosition=[0 0 screenposition(3:4)], PaperSize=[screenposition(3:4)]);
    exportgraphics(gcf, fullfile('figures', fname), 'ContentType', 'image', 'Resolution', 300);
end

