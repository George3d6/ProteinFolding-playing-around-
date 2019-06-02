using Bio.Structure: structure, coords, downloadpdb, PDB, read, resname, atoms, resnumber, serial, standardselector, AminoAcidSequence, inscode, atomname, cbetaselector, collectatoms,ContactMap
using Bio.Structure
using Flux
using Flux: @epochs
using Statistics
using Plots: plot, display


include("resnet.jl")

function get_protein_names()
    #["1CNL"]
    map((x) -> "1CNL", zeros(1))
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


function main()
    atom_dict = Dict{String,Int32}()
    atom_dict["NULL"] = 0
    last_atom_index = 0

    residue_dict = Dict{String,Int32}()
    residue_dict["NULL"] = 0
    last_residue_index = 0

    download_proteins()

    a_model_residue_onehot_names_arr = []
    a_model_residue_distances_arr = []

    min_distance = 10^62
    max_distance = -10^62

    for protein_name in get_protein_names()
        protein = read("protein_data/$protein_name.pdb", PDB)

        a_model_residue_names = []
        a_model_residue_distances = []

        contacts = ContactMap(collectatoms(protein["A"], cbetaselector), 8.0)
        contact_data = contacts.data
        #display(plot(contacts))

        '''
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
        '''

        onehot_names = Flux.onehotbatch(a_model_residue_names,a_model_residue_names)
        onehot_names_normal = []
        for b in onehot_names
            push!(onehot_names_normal, b)
        end

        push!(a_model_residue_onehot_names_arr, onehot_names_normal)
        push!(a_model_residue_distances_arr, a_model_residue_distances)
    end

    normalized_a_model_residue_distances_arr = map((x) -> (x .- min_distance) ./ (max_distance - min_distance), a_model_residue_distances_arr)

    max_in_len = length(a_model_residue_onehot_names_arr[1])
    max_out_len = length(normalized_a_model_residue_distances_arr[1])
    println(max_in_len)
    println(max_out_len)


    model = Flux.Chain(
        Dense(max_in_len,max_in_len),
        Dense(max_in_len,max_in_len),
        LSTM(max_in_len, UInt32(max_in_len/2)),
        LSTM( UInt32(max_in_len/2),  UInt32(max_in_len/2)),
        Dense( UInt32(max_in_len/2),max_out_len),
        softmax
    )

    function loss(x, y)
        Flux.reset!(model)
        Flux.mse(model(x), y)
    end

    function single_ele_acc(y1, y2)
        diff = abs(y1 - y2)
        if diff > 0.1
            return 0
        else
            return (1 - abs(y1 - y2)) ^ 100
        end
    end

    accuracy(y, y_test) = mean(single_ele_acc.(y,y_test))

    a_model_residue_onehot_names_arr = Tracker.data(a_model_residue_onehot_names_arr)
    normalized_a_model_residue_distances_arr = Tracker.data(normalized_a_model_residue_distances_arr)

    dataset = zip(a_model_residue_onehot_names_arr, normalized_a_model_residue_distances_arr)

    @epochs 12 Flux.train!(loss, params(model), dataset, ADAM())

    predictions = model(a_model_residue_onehot_names_arr[1])
    acc = accuracy(predictions, normalized_a_model_residue_distances_arr[1])

    println(predictions)
    println(normalized_a_model_residue_distances_arr[1])
    println(acc)

    return

    output_length = 2460
    for protein_name in get_protein_names()
        protein = read("protein_data/$protein_name.pdb", PDB)

        locations_arr = zeros(Float32, output_length)
        atoms_arr = zeros(Int32, UInt32(output_length/3))

        for residue in protein["A"]
            name = resname(residue)
            for atom in atoms(residue)
                location = coords(atom[2])
                index = serial(atom[2])
                locations_arr[index*3 - 2] = location[1] * 10^-2
                locations_arr[index*3 - 1] = location[2] * 10^-2
                locations_arr[index*3] = location[3] * 10^-2

                atom_name = atomname(atom[2])
                atom_index = get(atom_dict, atom_name, false)
                if atom_index == false
                    atom_index = last_atom_index
                    last_atom_index += 1
                    atom_dict[atom_name] = atom_index
                end
                atoms_arr[index] = atom_index
            end
        end
        seq = AminoAcidSequence(protein["A"], standardselector)
        seq = String(seq)
        index_seq = zeros(Int32,length(seq))

        for index in 1:length(seq)
            residue_name = string(seq[index])
            residue_index = get(residue_dict, residue_name, false)
            if residue_index == false
                residue_index = last_residue_index
                last_residue_index += 1
                residue_dict[residue_name] = residue_index
            end
            index_seq[index] = residue_index
        end
    end
end
