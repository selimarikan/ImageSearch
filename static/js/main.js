// Somehow dropzone auto discover does not register the events so use
//  manual registration
Dropzone.autoDiscover = false;

$(document).ready(function () {
    console.log("ready!");

    let imageDropzone = $("#imageDropzone");

    function CreateCard(imageB64, distance) {
        let newPath = "data:image/jpeg;base64, " + imageB64;
        let div = `
        <div class="col">
            <div class="card mb-3 result-card">
            <img src="` + newPath + `" class="card-img-top result-img" alt="result">
                <div class="card-body">
                    <h5 class="card-title">Result</h5>
                    <p class="card-text"> </p>
                    <p class="card-text"><small class="text-muted">Distance: ` + distance + `</small></p>
                </div>
            </div>
        </div>
        `
        return div;
    }

    function ShowResults(file, response) {
        let resultContainer = $("#result-card-container");

        // Remove old results first
        resultContainer.empty();

        for (let index = 0; index < response["imgs"].length; index++) {
            let divHTML = CreateCard(response["imgs"][index], response["distances"][index]);
            resultContainer.append(divHTML);
        }

    }
    Dropzone.options.imageDropzone = {
        init: function () {
            this.on("success", ShowResults);
        },
    };

    imageDropzone.dropzone();
});