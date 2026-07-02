function m = mark(i,j,k)
lines = {"-",":",""};
markers = {"", "-", "o", "x", "d", ".", "+", "s", "p", "h", "^", "v", ">", "<"};
colors = {"k", "c", "r", "b", "g", "m"};
m = lines{i} + markers{j} + colors(k);
end