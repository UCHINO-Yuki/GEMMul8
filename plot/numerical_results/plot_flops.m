function plot_flops(gpu)

arguments (Input)
    gpu (1,1) string = "GB200"
end

FontSize = 8;
type_in = "d";

%% get data
dir_name = dir(gpu + "/oz2_results_" + type_in + "gemm_time*");
file_name = gpu + "/" + dir_name.name;
data = detectImportOptions(file_name);

idx = find(strcmp(data.VariableNames,"m"));
data.SelectedVariableNames = idx;
data_m = readmatrix(file_name,data);

idx = find(strcmp(data.VariableNames,"k"));
data.SelectedVariableNames = idx;
data_k = readmatrix(file_name,data);

idx = find(contains(data.VariableNames,"unction"));
data.SelectedVariableNames = idx;
data_func = readmatrix(file_name,data);

idx = find(contains(data.VariableNames,"TFLOPS"));
data.SelectedVariableNames = idx;
data_tflops = readmatrix(file_name,data);

if contains(gpu,"RTX")
    idx = data_m < 16384;
    data_m = data_m(idx);
    data_k = data_k(idx);
    data_func = data_func(idx);
    data_tflops = data_tflops(idx);
end

%% plot
size_list = [1024 2048 4096 8192 16384 32768];
fig = figure('Position',[50,50,550,450]);
for i=length(size_list):-1:1
    if size_list(i)>max(data_m(contains(data_func,"OS2-i8-fast-15")))
        size_list(i)=[];
    end
end
if length(size_list)<=4
    t = tiledlayout(1,max(4,ceil(length(size_list)/2)));
    fig.Position(4) = 245;
else
    t = tiledlayout(2,max(3,ceil(length(size_list)/2)));
end
xlims_max = [inf,0];
i8_f8_ratio = [inf 0];
yl = [inf,0];
for tid = 1:length(size_list)
    m = size_list(tid);
    nexttile; hold on; grid on;
    xlims = [];

    idx = strcmp(data_func, "DGEMM") & data_m == m;
    if any(idx)
        tflops = data_tflops(idx);
        k = data_k(idx);
        plot(1:length(k),tflops,mark(3,1,1),'DisplayName',"native FP64 DGEMM", 'MarkerSize',5, 'LineWidth',1);
        tflops_DGEMM = tflops;
    end

    idx = strcmp(data_func, "OS1-7") & data_m == m;
    if any(idx)
        tflops = data_tflops(idx);
        k = data_k(idx);
        plot(1:length(k),tflops,mark(1,1,2),'DisplayName',"INT8 Ozaki-I (7 slices)", 'MarkerSize',5, 'LineWidth',1);
        tflops_i8oz1 = tflops;
    end

    idx = strcmp(data_func, "OS1-11") & data_m == m;
    if any(idx)
        tflops = data_tflops(idx);
        k = data_k(idx);
        plot(1:length(k),tflops,mark(2,1,2),'DisplayName',"INT8 Ozaki-I (11 slices)", 'MarkerSize',5, 'LineWidth',1);
        tflops_f8oz1 = tflops;
    end

    idx = strcmp(data_func, "OS2-i8-fast-15") & data_m == m;
    if any(idx)
        tflops = data_tflops(idx);
        k = data_k(idx);
        plot(1:length(k),tflops,mark(1,3,3),'DisplayName',"INT8 Ozaki-II fast (15 moduli)", 'MarkerSize',5, 'LineWidth',1);
        tflops_i8fast = tflops;
    end

    idx = strcmp(data_func, "OS2-i8-accu-15") & data_m == m;
    if any(idx)
        tflops = data_tflops(idx);
        k = data_k(idx);
        plot(1:length(k),tflops,mark(1,4,4),'DisplayName',"INT8 Ozaki-II accu. (15 moduli)", 'MarkerSize',5, 'LineWidth',1);
        tflops_i8accu = tflops;
    end

    xlims = k;

    if m==16384
        tflops_i8fast(k==16384),tflops_i8accu(k==16384)
    end
    m
    i8_dgemm = [tflops_i8fast,tflops_i8accu]./tflops_DGEMM(1:length(k))
    [i8_dgemm_min,i8_dgemm_max] = bounds(i8_dgemm,'all')

    idx = strcmp(data_func, "OS2-f8-fast-12") & data_m == m;
    if any(idx)
        tflops = data_tflops(idx);
        k = data_k(idx);
        plot(1:length(k),tflops,mark(2,3,5),'DisplayName',"FP8 Ozaki-II fast (12 moduli)", 'MarkerSize',5, 'LineWidth',1);
        tflops_f8fast = tflops;
    end

    idx = strcmp(data_func, "OS2-f8-accu-12") & data_m == m;
    if any(idx)
        tflops = data_tflops(idx);
        k = data_k(idx);
        plot(1:length(k),tflops,mark(2,4,6),'DisplayName',"FP8 Ozaki-II accu. (12 moduli)", 'MarkerSize',5, 'LineWidth',1);
        tflops_f8accu = tflops;
    end

    f8_dgemm = [tflops_f8fast,tflops_f8accu]./tflops_DGEMM(1:length(k))
    [f8_dgemm_min,f8_dgemm_max] = bounds(f8_dgemm,'all')

    if m==16384
        tflops_f8fast(k==16384),tflops_f8accu(k==16384)
    end

    i8_f8 = [tflops_i8fast(1:length(k)),tflops_i8accu(1:length(k)),tflops_i8accu(1:length(k)),tflops_i8fast(1:length(k))]...
        ./[tflops_f8fast,tflops_f8accu,tflops_f8fast,tflops_f8accu];
    [i8_f8_min,i8_f8_max] = bounds(i8_f8,'all')
    i8_f8_ratio(1) = min(i8_f8_ratio(1),i8_f8_min);
    i8_f8_ratio(2) = max(i8_f8_ratio(2),i8_f8_max);

    xlims_max(1) = min(xlims_max(1), min(k));
    xlims_max(2) = max(xlims_max(2), max(k));

    title("{\itm=n=" + m +"=2^{" + log2(m) + "}}");
    set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');
    ylim('padded');
    yl_tmp = ylim;
    yl(1) = min(yl(1), yl_tmp(1));
    yl(2) = max(yl(2), yl_tmp(2));
    xlims = pow2(log2(xlims_max(1)):log2(xlims_max(2)));
    xlim([1 length(xlims)]);
    xticks(1:length(xlims));
    xticklabels("2^{" + log2(xlims) + "}")
    xtickangle(0)
end

inc = ceil((yl(2)-yl(1))/45)*5;
if (yl(2)-yl(1)) >= 160
    inc = 20;
end
for tid = 1:length(size_list)
    nexttile(tid);
    ylim(yl);
    yticks(0:inc:300);
end

i8_f8_ratio

nexttile(1);
lgd = legend('Interpreter','tex','FontName','Yu Gothic UI Semibold','IconColumnWidth',15,'NumColumns',2);
lgd.Layout.Tile = 'north';
t.TileSpacing = "tight";
t.Padding = "compact";
xlabel(t,"\itk", 'Interpreter', 'tex','FontName','Yu Gothic UI Semibold');
ylabel(t,'TFLOP/s','FontName','Yu Gothic UI Semibold');
set(gca,'FontSize',FontSize,'FontName','Yu Gothic UI Semibold');

savefig(fig,gpu+"/"+gpu+"_flops_"+type_in);
exportgraphics(fig,gpu+"/"+gpu+"_flops_"+type_in+".png",'Resolution',600);
end

%%
function m = mark(i,j,k)
lines = {"-",":","--",""};
markers = {"", "o", "x", "d", "p", "+", ".", "s", "h", "^", "v", ">", "<"};
colors = {"k", "c", "r", "b", "g", "m"};
m = lines{i} + markers{j} + colors(k);
end
