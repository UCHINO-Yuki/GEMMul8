function plot_accuracy(type_in,GPU_name)

arguments (Input)
    type_in (1,1) string = "d"
    GPU_name (1,1) string = "RTX5080"
end

FontSize = 8;

%% get data
dir_name_i8  = dir(GPU_name + "/oz2_results_i_" + type_in + "_accuracy*");
if ~isempty(dir_name_i8)
    file_name_i8 = GPU_name + "/" + dir_name_i8.name;
    data_i8 = detectImportOptions(file_name_i8);
    data_i8.SelectedVariableNames = 1;
    phi_i8 = readmatrix(file_name_i8,data_i8);
    phi_i8 = phi_i8(2:end,:);
    data_i8.SelectedVariableNames = 2;
    k_i8 = readmatrix(file_name_i8,data_i8);
    k_i8 = k_i8(2:end,:);
    data_i8.SelectedVariableNames = 3;
    func_i8 = readmatrix(file_name_i8,data_i8);
    func_i8 = func_i8(2:end,:);
    data_i8.SelectedVariableNames = 4:length(data_i8.VariableNames);
    err_i8 = readmatrix(file_name_i8,data_i8);
    moduli_i8 = err_i8(1,:);
    err_i8 = err_i8(2:end,:);
end

dir_name_f8  = dir(GPU_name + "/oz2_results_f_" + type_in + "_accuracy*");
if ~isempty(dir_name_f8)
    file_name_f8 = GPU_name + "/" + dir_name_f8.name;
    data_f8 = detectImportOptions(file_name_f8);
    data_f8.SelectedVariableNames = 1;
    phi_f8 = readmatrix(file_name_f8,data_f8);
    phi_f8 = phi_f8(2:end,:);
    data_f8.SelectedVariableNames = 2;
    k_f8 = readmatrix(file_name_f8,data_f8);
    k_f8 = k_f8(2:end,:);
    data_f8.SelectedVariableNames = 3;
    func_f8 = readmatrix(file_name_f8,data_f8);
    func_f8 = func_f8(2:end,:);
    data_f8.SelectedVariableNames = 4:length(data_f8.VariableNames);
    err_f8 = readmatrix(file_name_f8,data_f8);
    moduli_f8 = err_f8(1,:);
    err_f8 = err_f8(2:end,:);
end

%% plot
yl_min = inf;
yl_max = 0;
phi = [-1, 1, 2, 4];
fig = figure('Position',[50,50,500,400]);
t = tiledlayout(2,2);
for tid = 1:4
    nexttile; hold on; grid on;

    for k = [1024, max(k_i8)]
        if ~isempty(dir_name_i8)
    
            idx = contains(func_i8,"DGEMM") & k_i8 == k & phi_i8 == phi(tid);
            err = err_i8(idx,:);
            plot(moduli_i8, err, mark(1,2-(k==1024),1), 'DisplayName', "native FP64 DGEMM {\itk=" + k +"}", 'MarkerSize',5, 'LineWidth',1);
    
            idx = contains(func_i8,"OS1-7") & k_i8 == k & phi_i8 == phi(tid);
            err = err_i8(idx,:);
            plot(moduli_i8, err, mark(1,2-(k==1024),2), 'DisplayName', "INT8-based Ozaki-I (7 slices) {\itk=" + k +"}", 'MarkerSize',5, 'LineWidth',1);
    
            idx = contains(func_i8,"OS2-fast") & k_i8 == k & phi_i8 == phi(tid);
            err = err_i8(idx,:);
            plot(moduli_i8, err, mark(1,2-(k==1024),3), 'DisplayName', "INT8-based Ozaki-II (fast) {\itk=" + k +"}", 'MarkerSize',5, 'LineWidth',1);
    
            idx = contains(func_i8,"OS2-accu") & k_i8 == k & phi_i8 == phi(tid);
            err = err_i8(idx,:);
            plot(moduli_i8, err, mark(1,2-(k==1024),4), 'DisplayName', "INT8-based Ozaki-II (acc.) {\itk=" + k +"}", 'MarkerSize',5, 'LineWidth',1);
    
            xlim([min(moduli_i8), max(moduli_i8)]);
            xticks(moduli_i8);
        end
    
        if ~isempty(dir_name_f8)
            idx = contains(func_f8,"OS2-fast") & k_f8 == k & phi_f8 == phi(tid);
            err = err_f8(idx,:);
            plot(moduli_f8, err, mark(1,2-(k==1024),5), 'DisplayName', "FP8-based Ozaki-II (fast) {\itk=" + k +"}", 'MarkerSize',5, 'LineWidth',1);
    
            idx = contains(func_f8,"OS2-accu") & k_f8 == k & phi_f8 == phi(tid);
            err = err_f8(idx,:);
            plot(moduli_f8, err, mark(1,2-(k==1024),6), 'DisplayName', "FP8-based Ozaki-II (acc.) {\itk=" + k +"}", 'MarkerSize',5, 'LineWidth',1);
    
            xlim([min(moduli_f8), max(moduli_f8)]);
            xticks(moduli_f8);
        end
    end

    if phi(tid)<0
        title("Normal dist. w/ mean 0 and std. dev. 1", 'Interpreter','tex');
    else
        title("{\it\phi = " + phi(tid) + "}", 'Interpreter','tex');
    end
    set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold','YScale','Log');
    ylim('padded');
    yticks(10.^(-16:4:16));
end

lgd = legend('Interpreter','tex','FontName','Yu Gothic UI Semibold','IconColumnWidth',15,'NumColumns',2);
lgd.Layout.Tile = 'north';
t.TileSpacing = "tight";
t.Padding = "compact";
xlabel(t,"Number of moduli");
ylabel(t,"Max. relative error");
set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');

savefig(fig,GPU_name+"/"+GPU_name+"_accuracy_"+type_in);
exportgraphics(fig,GPU_name+"/"+GPU_name+"_accuracy_"+type_in+".png",'Resolution',600);
end
