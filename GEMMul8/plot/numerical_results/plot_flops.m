function plot_flops(type_in,GPU_name)

arguments (Input)
    type_in (1,1) string = "d"
    GPU_name (1,1) string = "RTX5080"
end

FontSize = 8;

%% get data
dir_name_i8  = dir(GPU_name + "/oz2_results_i_" + type_in + "_time*");
m_i8 = [];
if ~isempty(dir_name_i8)
    file_name_i8 = GPU_name + "/" + dir_name_i8.name;
    data_i8 = detectImportOptions(file_name_i8);
    data_i8.SelectedVariableNames = 2;
    m_i8 = readmatrix(file_name_i8,data_i8);
    data_i8.SelectedVariableNames = 4;
    k_i8 = readmatrix(file_name_i8,data_i8);
    data_i8.SelectedVariableNames = 5;
    func_i8 = readmatrix(file_name_i8,data_i8);
    data_i8.SelectedVariableNames = 8;
    tflops_i8 = readmatrix(file_name_i8,data_i8);
end

m_f8 = [];
dir_name_f8  = dir(GPU_name + "/oz2_results_f_" + type_in + "_time*");
if ~isempty(dir_name_f8)
    file_name_f8 = GPU_name + "/" + dir_name_f8.name;
    data_f8 = detectImportOptions(file_name_f8);
    data_f8.SelectedVariableNames = 2;
    m_f8 = readmatrix(file_name_f8,data_f8);
    data_f8.SelectedVariableNames = 4;
    k_f8 = readmatrix(file_name_f8,data_f8);
    data_f8.SelectedVariableNames = 5;
    func_f8 = readmatrix(file_name_f8,data_f8);
    data_f8.SelectedVariableNames = 8;
    tflops_f8 = readmatrix(file_name_f8,data_f8);
end

%% plot
size_list = [1024 2048 4096 8192 16384 32768];
fig = figure('Position',[50,50,550,350]);
for i=length(size_list):-1:1
    if size_list(i)>min(max(m_i8),max(m_f8))
        size_list(i)=[];
    end
end
t = tiledlayout(2,ceil(length(size_list)/2));
for tid = 1:length(size_list)
    m = size_list(tid);
    nexttile; hold on; grid on;
    xlims = [];

    if ~isempty(dir_name_i8)
        idx = contains(func_i8,"DGEMM") & m_i8 == m;
        if any(idx)
            tflops = tflops_i8(idx);
            k = k_i8(idx);
            plot(1:length(k),tflops,mark(1,1,1),'DisplayName',"native FP64 DGEMM", 'MarkerSize',5, 'LineWidth',1);
            tflops_DGEMM = tflops;
        end

        idx = contains(func_i8,"fast-16") & m_i8 == m;
        if any(idx)
            tflops = tflops_i8(idx);
            k = k_i8(idx);
            plot(1:length(k),tflops,mark(1,1,2),'DisplayName',"INT8-based Ozaki-II fast (16 moduli)", 'MarkerSize',5, 'LineWidth',1);
            tflops_i8fast = tflops;
        end

        idx = contains(func_i8,"accu-15") & m_i8 == m;
        if any(idx)
            tflops = tflops_i8(idx);
            k = k_i8(idx);
            plot(1:length(k),tflops,mark(1,1,3),'DisplayName',"INT8-based Ozaki-II accu. (15 moduli)", 'MarkerSize',5, 'LineWidth',1);
            tflops_i8accu = tflops;
        end

        xlims = k;

        if m==16384
            tflops_i8fast(k==16384),tflops_i8accu(k==16384)
        end
        % m
        % i8_dgemm = [tflops_i8fast,tflops_i8accu]./tflops_DGEMM
    end

    if ~isempty(dir_name_f8)
        idx = contains(func_f8,"fast-13") & m_f8 == m;
        if any(idx)
            tflops = tflops_f8(idx);
            k = k_f8(idx);
            plot(1:length(k),tflops,mark(1,1,4),'DisplayName',"FP8-based Ozaki-II fast (13 moduli)", 'MarkerSize',5, 'LineWidth',1);
            tflops_f8fast = tflops;
        end

        idx = contains(func_f8,"accu-12") & m_f8 == m;
        if any(idx)
            tflops = tflops_f8(idx);
            k = k_f8(idx);
            plot(1:length(k),tflops,mark(1,1,5),'DisplayName',"FP8-based Ozaki-II accu. (12 moduli)", 'MarkerSize',5, 'LineWidth',1);
            tflops_f8accu = tflops;
        end

        % f8_dgemm = [tflops_f8fast,tflops_f8accu]./tflops_DGEMM(1:length(k))
        % i8_f8 = [tflops_i8fast(1:length(k)),tflops_i8accu(1:length(k)),tflops_i8accu(1:length(k)),tflops_i8fast(1:length(k))]...
        %     ./[tflops_f8fast,tflops_f8accu,tflops_f8fast,tflops_f8accu];
        % [i8_f8_min,i8_f8_max] = bounds(i8_f8,'all')

        if m==16384
            tflops_f8fast(k==16384),tflops_f8accu(k==16384)
        end

        if length(k) > length(xlims)
            xlims = k;
        end
    end

    title("{\itm=n=" + m +"=2^{" + log2(m) + "}}");
    set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');
    ylim('padded');
    xlim([1 length(xlims)]);
    xticks(1:length(xlims));
    xticklabels("2^{" + log2(xlims) + "}")
    xtickangle(0)
end

nexttile(1);
lgd = legend('Interpreter','tex','FontName','Yu Gothic UI Semibold','IconColumnWidth',15,'NumColumns',2);
lgd.Layout.Tile = 'north';
t.TileSpacing = "tight";
t.Padding = "compact";
xlabel(t,"\itk");
ylabel(t,"TFLOP/s ({\it2mnk/sec/10^{12}})");
set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');

savefig(fig,GPU_name+"/"+GPU_name+"_flops_"+type_in);
exportgraphics(fig,GPU_name+"/"+GPU_name+"_flops_"+type_in+".png",'Resolution',600);
end
