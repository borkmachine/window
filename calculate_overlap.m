% return m1 union m2 devided by m1
function percentage = calculate_overlap(mask1, mask2)
    overlap = mask1.*mask2;
    percentage = sum(sum(overlap))/sum(sum(mask1));
    return
end