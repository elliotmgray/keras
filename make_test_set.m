raw_img = '/Users/grael/Desktop/melanoma_epidermis/raw/image';
raw_label = '/Users/grael/Desktop/melanoma_epidermis/raw/label';

test_image_dir = '/Users/grael/Desktop/melanoma_epidermis/test/image';
test_result_dir = '/Users/grael/Desktop/melanoma_epidermis/test/result';
model_dir = '/Users/grael/Desktop/melanoma_epidermis/model';

crop_size = 256;

raw_names = {
    '70901-64576'
    '71001-64408'
    '71288-64806'
    '71357-64285'
    '71446-63955'
};

for l = 1:length(raw_names)
    if ~ isdir([test_image_dir '/loo_' raw_names{l}])
        mkdir([test_image_dir '/loo_' raw_names{l}]);
    end
end
for l = 1:length(raw_names)
    if ~ isdir([test_result_dir '/loo_' raw_names{l}])
        mkdir([test_result_dir '/loo_' raw_names{l}]);
    end
end
for l = 1:length(raw_names)
    if ~ isdir([model_dir '/loo_' raw_names{l}])
        mkdir([model_dir '/loo_' raw_names{l}]);
    end
end

for i = 1:length(raw_names)

    img = imread([raw_img '/' raw_names{i} '.tif']);
    x_steps = floor(size(img, 1)/crop_size);
    y_steps = floor(size(img, 2)/crop_size);
    count = 0;
    for x = 1:x_steps
        for y = 1:y_steps

            x0 = (x-1)*crop_size + 1;
            x1 = x*crop_size;
            y0 = (y-1)*crop_size + 1;
            y1 = y*crop_size;

            imwrite(img(x0:x1, y0:y1, 1:3),...
                [test_image_dir '/loo_' raw_names{i} '/' num2str(count, '%06d') '.tif'],...
                'writemode', 'overwrite');
            count = count+1;
        end
    end
end
