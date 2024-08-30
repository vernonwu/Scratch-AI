class PyTorchCodeGenerator {
    constructor(graph, parentNodes) {
        this.graph = graph;
        this.parentNodes = parentNodes;
        this.layerNames = {};
        this.generateModuleName();
    }

    generateModuleName(){
        let moduleCount = {};
        for (let i = 0; i < this.graph.nodeList.length; i++) {
            let node = this.graph.nodeList[i];
            let layerName = node.graphName;
            if (!moduleCount[layerName]) {
                moduleCount[layerName] = 1;
            } else {
                moduleCount[layerName]++;
            }
            this.layerNames[i] = layerName + moduleCount[layerName];
        }
    }

    generatePythonLine(index){
        let layerName = this.layerNames[index];
        let parent = this.parentNodes[index];
        let node = this.graph.nodeList[index];
        let parentNames = parent.map(p => this.layerNames[p]);
        if (parent.length > 1) {
            return `        ${layerName}_output = torch.cat([${parentNames.map(p => p + '_output').join(', ')}], dim=${node.parameterList.dim})\n`;
        }
        else {
            return `        ${layerName}_output = self.${layerName}(${parentNames[0] + '_output'})\n`;
        }
    }

    generate() {
        let code = "import torch\nimport torch.nn as nn\n\n";
        code += "class MyModel(nn.Module):\n";
        code += "    def __init__(self):\n";
        code += "        super(MyModel, self).__init__()\n";

        // Define layers
        for (let i = 1; i < this.graph.nodeList.length-1; i++) {
            let node = this.graph.nodeList[i];
            let layerType = node.graphName;
            if (layerType !== 'concatenate') {
                let layerName = this.layerNames[i];
                let layerParams = node.parameterList;
                let layerParamsStr = JSON.stringify(layerParams);
                code += `        self.${layerName} = nn.${layerType}(${layerParamsStr})\n`;
            }
        }

        // Define forward pass
        code += "\n    def forward(self, x):\n";
        code += "        input1_output = x \n";

        for (let i = 1; i < this.graph.nodeList.length - 1; i++) {
            let node = this.graph.nodeList[i];
            code += this.generatePythonLine(i);
        }

        let outputLayer = this.layerNames[this.graph.nodeList.length - 2];

        code += `        return ${outputLayer}_output\n`;
        return code;
    }
}