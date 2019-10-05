using Bio.Structure: structure, coords, downloadpdb, PDB, read, resname, atoms, resnumber, serial, standardselector, AminoAcidSequence, inscode, atomname, cbetaselector, collectatoms,ContactMap
using Bio.Structure
using Flux
using Flux: @epochs
using Statistics
using Plots: plot, display,heatmap


include("resnet.jl")

function get_protein_names()
    #["1CNL"]
    map((x) -> "1CNL", zeros(500))
end

function download_proteins()
    protein_dataset = get_protein_names()
    for protein_name in protein_dataset
        if !isfile("protein_data/$protein_name.pdb")
            println("Downloading protein: $protein_name !")
            downloadpdb(protein_name, pdb_dir="protein_data")
        end
    end
end

function oh_encode_residues(residues_arr)
    unique_residue_names = [Set(collect(Iterators.flatten(residues_arr)))...]
    encoded_residues_arr = map((x) -> Flux.onehotbatch(x,unique_residue_names),residues_arr)
    return encoded_residues_arr
end



function main()
    atom_dict = Dict{String,Int32}()
    atom_dict["NULL"] = 0
    last_atom_index = 0

    residue_dict = Dict{String,Int32}()
    residue_dict["NULL"] = 0
    last_residue_index = 0

    download_proteins()

    residues_arr = []
    contacts_arr = []

    min_distance = 10^62
    max_distance = -10^62

    for protein_name in get_protein_names()
        protein = read("protein_data/$protein_name.pdb", PDB)

        residues = map((x) -> String(resname(x)), protein["A"])
        push!(residues_arr, residues)
        contacts = ContactMap(collectatoms(protein["A"], cbetaselector), 8.0)
        contact_data = contacts.data
        push!(contacts_arr, contact_data)

        """
        for i in 1:length(protein["A"]) - 1
            push!(a_model_residue_names, String(resname(protein["A"][i])))
            if i > 1
                distance = BioStructures.distance(protein["A"][i-1],protein["A"][i])
                push!(a_model_residue_distances, distance)
                if distance < min_distance
                    min_distance = distance
                end
                if distance > max_distance
                    max_distance = distance
                end
            end
        end
        """
    end

    e_residues_arr = oh_encode_residues(residues_arr)

    #e_residues_arr = map((x) -> cat(dims=4,x),e_residues_arr)
    e_residues_arr =  map((x) -> [x...],e_residues_arr)
    contacts_arr =  map((x) -> [x...],contacts_arr)

    max_in_len = length(e_residues_arr[1])
    max_out_len = length(contacts_arr[1])

    println("Maximum input size is: $max_in_len !\nMaximum output size is: $max_out_len !")

    model = ResNet101()

    function loss(x, y)
        Flux.reset!(model)
        Flux.mse(model(x), y)
    end

    single_ele_acc(y1, y2) = UInt32(y1 == y2)

    accuracy(y, y_test) = mean(single_ele_acc.(y,y_test))

    dataset = zip(e_residues_arr, contacts_arr)

    rspr = reshape(contacts_arr[1],(UInt32(length(contacts_arr[1])/12),12))

    @epochs 22 Flux.train!(loss, params(model), dataset, ADAM())

    predictions = model(e_residues_arr[1])
    function n_to_bool(x)
        if x < 0.5
            return false
        end
            return true
    end
    predictions_binary = map(n_to_bool, predictions)
    acc = accuracy(predictions_binary, contacts_arr[1])
    println("Got a total accuracy of: $acc for predicting the C-beta atoms distance matrix")

    rsp = reshape(predictions,(UInt32(length(predictions)/12),12))
    display(heatmap(rsp))
end
# Tracker.data(arr)
