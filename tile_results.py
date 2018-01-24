raw_img = '/Users/grael/Desktop/melanoma_epidermis/raw/image';
raw_label = '/Users/grael/Desktop/melanoma_epidermis/raw/label';

test_image = '/Users/grael/Desktop/melanoma_epidermis/test/image';
test_result = '/Users/grael/Desktop/melanoma_epidermis/test/result';

crop_size = 256;

raw_names = {
    '70901-64576'
    '71001-64408'
    '71288-64806'
    '71357-64285'
    '71446-63955'
};
map = colormap(parula(256));

for i = 1:length(raw_names)
    
    img = imread([raw_img '/' raw_names{i} '.tif']);
    label = imread([raw_label '/' raw_names{i} '.tif']);
    x_steps = floor(size(img, 1)/crop_size);
    y_steps = floor(size(img, 2)/crop_size);
    
    result_tile = single(zeros(x_steps*crop_size, y_steps*crop_size));
    label_tile = zeros(x_steps*crop_size, y_steps*crop_size);
    test_tile = uint8(zeros(x_steps*crop_size, y_steps*crop_size, 3));
    
    count = 0;
    for x = 1:x_steps
        for y = 1:y_steps
            
            x0 = (x-1)*crop_size + 1;
            x1 = x*crop_size;
            y0 = (y-1)*crop_size + 1;
            y1 = y*crop_size;
            
            
            result_i = imread([test_result '/loo_' raw_names{i} '/' num2str(count) '.tif']);
            test_i = imread([test_image '/loo_' raw_names{i} '/' num2str(count, '%06d') '.tif']);

            result_tile(x0:x1, y0:y1) = result_i;
            label_tile(x0:x1, y0:y1) = label(x0:x1, y0:y1);
            test_tile(x0:x1, y0:y1, :) = test_i;
            
            count = count + 1;
        end

    end
    figure
    
    subplot(2, 1, 1);
    imshow(test_tile)
    hold on;
    visboundaries(label_tile)
    
    subplot(2, 1, 2);
    imshow(result_tile, map);
    hold on;
    visboundaries(label_tile)
    
end
