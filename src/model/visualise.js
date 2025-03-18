/**
 * Draw a grayscaled 28x28 image onto the canvas.
 * 
 * @param {[number]} pixels The flat array of pixels of the image, instead of rgba values only a single value
 * @param {HTMLCanvasElement} canvas The html dom canvas element to render image onto
 */
async function renderImage(pixels, canvas) {
    const ctx = canvas.getContext("2d");
    
    ctx.imageSmoothingEnabled = false;
    
    // Set the canvas size to match the image (28x28 pixels)
    canvas.width = 28;
    canvas.height = 28;
    
    const imageData = ctx.createImageData(28, 28);
    
    for (let i = 0; i < pixels.length; i++) {
        imageData.data[i * 4] = pixels[i];     // Red channel (greyscale image)
        imageData.data[i * 4 + 1] = pixels[i]; // Green channel (greyscale image)
        imageData.data[i * 4 + 2] = pixels[i]; // Blue channel (greyscale image)
        imageData.data[i * 4 + 3] = 255;       // Alpha channel (fully opaque)
    }
    
    // Put the image data on the canvas
    ctx.putImageData(imageData, 0, 0);
}

export { renderImage };