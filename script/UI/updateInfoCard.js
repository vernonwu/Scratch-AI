function updateInfoCard(layer) {
    const cardTitle = document.getElementById('cardtitle');
    const cardText = document.querySelector('.card-text');
    const cardImg = document.getElementById('cardimg');

    switch(layer) {
        case 'conv2d':
            cardTitle.innerText = 'Conv2D Layer';
            cardText.innerText = 'Applies a 2D convolution over an input signal composed of several input planes.';
            cardImg.src = 'image/conv2d.png';
            break;
        case 'FC':
            cardTitle.innerText = 'Fully Connected Layer';
            cardText.innerText = 'Connects every neuron in one layer to every neuron in another layer.';
            cardImg.src = 'image/FC.png';
            break;
        case 'pooling':
            cardTitle.innerText = 'Pooling Layer';
            cardText.innerText = 'Reduces the spatial dimensions (width and height) of the input volume.';
            cardImg.src = 'image/pooling.png';
            break;
        case 'input':
            cardTitle.innerText = 'Input Layer';
            cardText.innerText = 'The first layer of the neural network that receives the input data.';
            cardImg.src = 'image/input.png';
            break;
        case 'output':
            cardTitle.innerText = 'Output Layer';
            cardText.innerText = 'The last layer of the neural network that produces the final output.';
            cardImg.src = 'image/output.png';
            break;
        default:
            cardTitle.innerText = 'Card title';
            cardText.innerText = 'NUll';
            cardImg.src = '...';
    }
}