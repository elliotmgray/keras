raw_img = '/Users/grael/Desktop/melanoma_epidermis/raw/image';
raw_label = '/Users/grael/Desktop/melanoma_epidermis/raw/label';

write_image = '/Users/grael/Desktop/melanoma_epidermis/train/image';
write_label = '/Users/grael/Desktop/melanoma_epidermis/train/label';
npy_dir = '/Users/grael/Desktop/melanoma_epidermis/npydata';

crop_size = 256;
crops_per_img = 100;

raw_names = {
    '70901-64576'
    '71001-64408'
    '71288-64806'
    '71357-64285'
    '71446-63955'};


% take the raw images and their paired labels. crop a bunch of 128x128
% regions out of them, with higher chance of taking them from the region
% that is positive for the label.

count = zeros(length(raw_names), crops_per_img);

for l = 1:length(raw_names)
    if ~ isdir([write_image '/loo_' raw_names{l}])
        mkdir([write_image '/loo_' raw_names{l}]);
    end
end
for l = 1:length(raw_names)
    if ~ isdir([write_label '/loo_' raw_names{l}])
        mkdir([write_label '/loo_' raw_names{l}]);
    end
end
for l = 1:length(raw_names)
    if ~ isdir([npy_dir '/loo_' raw_names{l}])
        mkdir([npy_dir '/loo_' raw_names{l}]);
    end
end

% leave-one-out cross-validation
for l = 1:length(raw_names)
    for i = 1:length(raw_names)
        if i ~= l
            img_i = imread([raw_img '/' raw_names{i} '.tif']);
            label_i = imread([raw_label '/' raw_names{i} '.tif']);

            assert(size(img_i, 3) == 4)

            img_i = img_i(:,:,1:3); % i dont know why they have a 4th channel.

        %     figure; imshow(img_i); hold on; visboundaries(imgaussfilt(label_i, 20));


            w = imgaussfilt(30*label_i + uint8(ones(size(label_i))), 10);
        %     figure; imagesc(w);

            w = double(w(:));
            w = w./sum(w);

            r = randsample(length(w), crops_per_img, true, w);

            [x, y] = ind2sub(size(label_i), r);

            x0 = x - crop_size / 2;
            y0 = y - crop_size / 2;
            x1 = x - 1 + crop_size / 2;
            y1 = y - 1 + crop_size / 2;

            neg_x = x0 <= 0;
            neg_y = y0 <= 0;
            pos_x = x1 >= size(img_i, 1);
            pos_y = y1 >= size(img_i, 2);

            x0(neg_x) = 1; x1(neg_x) = crop_size;
            y0(neg_y) = 1; y1(neg_y) = crop_size;
            x1(pos_x) = size(img_i, 1); x0(pos_x) = size(img_i, 1) - crop_size + 1;
            y1(pos_y) = size(img_i, 2); y0(pos_y) = size(img_i, 2) - crop_size + 1;

            for j = 1:crops_per_img
                imwrite(img_i(x0(j):x1(j), y0(j):y1(j), :), [write_image '/loo_' raw_names{l} '/' raw_names{i} '-' num2str(j-1) '.tif'], 'writemode', 'overwrite');
                imwrite(label_i(x0(j):x1(j), y0(j):y1(j)), [write_label '/loo_' raw_names{l} '/' raw_names{i} '-' num2str(j-1) '.tif'], 'writemode', 'overwrite');

                count(i,j) = nnz(label_i(x0(j):x1(j), y0(j):y1(j)))/numel(label_i(x0(j):x1(j), y0(j):y1(j)));
            end
        end
    end
    
    figure; hist(count(:))
    
end
