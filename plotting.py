import matplotlib.pyplot as plt
import os

def plot_results(original, sst, cells, seg_flat, filtered_sst, sst_cells, c, exp, fov):
    plt.figure(figsize=(16, 16))

    # Top left: Original Image
    plt.subplot(3, 2, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(3, 2, 2)
    plt.imshow(sst)
    plt.title('SST (Post-Color Deconvolution)')
    plt.axis('off')

    plt.subplot(3, 2, 3)
    plt.imshow(cells)
    plt.imshow(seg_flat, cmap='gist_gray', alpha=0.3)  # Positive cells in color overlay
    plt.title('Cells (Post-Color Deconvolution) with Cellpose Masks')
    plt.axis('off')

    plt.subplot(3, 2, 5)
    plt.imshow(seg_flat, cmap='gray', alpha=0.8)  # Original cell mask in gray
    plt.imshow(sst_cells, cmap='jet', alpha=0.6)  # Positive cells in color overlay
    plt.title("Somatostatin Positive Cells")
    plt.axis("off")

    plt.subplot(3, 2, 4)
    plt.imshow(filtered_sst, cmap='gray')
    plt.title('Thresholded SST')
    plt.axis('off')

    # Bottom right: Somatostatin Positive Cells
    plt.subplot(3, 2, 6)
    plt.imshow(original)  # Original cell mask in gray
    plt.imshow(sst_cells, cmap='gist_gray', alpha=0.5)  # Positive cells in color overlay
    plt.title("Somatostatin Positive Cells Overlay")
    plt.axis("off")
    
    x = exp.replace(".czi",'')
    filter_level = ""
    if c==0:
        filter_level = "high_filter"
    else:
        filter_level = "low_filter"
    
    directory = "./Results/"+x+"/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(directory+str(fov)+"_"+filter_level+".png", bbox_inches='tight', dpi=300)
    plt.close()