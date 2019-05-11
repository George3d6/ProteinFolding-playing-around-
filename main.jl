using Bio.Structure: coords, downloadpdb, PDB, read, resname, atoms, resnumber, serial, standardselector, AminoAcidSequence, inscode, atomname
using Flux
using Flux: @epochs
using Statistics

function get_protein_names()
    ["1EN2"]
end

function download_proteins()
    protein_dataset = get_protein_names()
    for protein_name in protein_dataset
        if !isfile("protein_data/$protein_name.pdb")
            downloadpdb(protein_name, pdb_dir="protein_data")
        end
    end
end


function main()
    atom_dict = Dict{String,Int32}()
    atom_dict["NULL"] = 0
    last_atom_index = 0

    download_proteins()
    output_length = 2601
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


        model = Chain(
          Dense(UInt32(output_length/3), UInt32(output_length), σ),
          LSTM(UInt32(output_length), UInt32(output_length)),
          Dense(UInt32(output_length), UInt32(output_length)),
          softmax)
         '''
        model = Chain(
            Dense(UInt32(output_length/3), UInt32(output_length), σ),
            Dense(UInt32(output_length), UInt32(output_length)),
            Dense(UInt32(output_length), UInt32(output_length)),
            softmax)
        '''

        function loss(x, y)
            Flux.reset!(model)
            Flux.mse(model(x), y)
        end

        accuracy(y, y_test) = mean(isapprox.(y, y_test))

        atoms_arr = Tracker.data(atoms_arr)
        locations_arr = Tracker.data(locations_arr)

        dataset = [(atoms_arr, locations_arr),(atoms_arr, locations_arr),(atoms_arr, locations_arr),(atoms_arr, locations_arr),(atoms_arr, locations_arr),(atoms_arr, locations_arr),(atoms_arr, locations_arr),(atoms_arr, locations_arr),(atoms_arr, locations_arr),(atoms_arr, locations_arr),(atoms_arr, locations_arr),(atoms_arr, locations_arr),(atoms_arr, locations_arr),(atoms_arr, locations_arr)]

        @epochs 30 Flux.train!(loss, params(model), dataset, ADAM())

        predictions = model(atoms_arr)

        println(predictions)
        println(locations_arr)

        acc = accuracy(predictions, locations_arr)
        println(acc)
    end
end
