function MP = op_kernel_cc_4n_rm(M, radius, thr)
%OP_CC_ Summary of this function goes here
%   Detailed explanation goes here
M_padded = padarray(M, [radius, radius], 'replicate', 'both');
[h, w] = size(M);

for i=1:h
    for j=1:w
        patch = M_padded(i:i+2*radius, j:j+2*radius);
        c_pixel = patch(radius+1, radius+1);
        mask = abs(patch-c_pixel) < thr;

        [labeled, num_labels] = bwlabel(mask, 4);

        for label_id=1:num_labels
            region=labeled==label_id;
            if any(region(1,:))||any(region(end,:))||any(region(:,1))||any(region(:,end))
                continue;
            end
            patch(region)=0;
        end

        M_padded(i:i+2*radius, j:j+2*radius) = patch;
    end
end
MP = M_padded(radius+1:end-radius, radius+1:end-radius);
end

