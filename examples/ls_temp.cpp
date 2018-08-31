#include "ls_temp.hpp"
#include "mars.hpp"

namespace mfem {
	class NDLSRefiner::RefinerImpl {
	public:
		using Mesh2 = mars::Mesh2;
		using Mesh3 = mars::Mesh3;
		using Mesh4 = mars::Mesh4;
		using Integer = mars::Integer;
		using Map = mars::Map;

		template<Integer Dim>
		using Elem = mars::Simplex<Dim, Dim>;

		template<Integer Dim>
		class Data {
		public:
			using Mesh = mars::Mesh<Dim, Dim>;
			using Bisection = mars::Bisection<Mesh>;
			using Map = mars::Map;
			
			Mesh mesh;
			Bisection bisection;
			std::shared_ptr<Map> map;
			bool is_initialized;


			Data()
			: mesh(), bisection(mesh), is_initialized(false)
			{}
		};

		RefinerImpl(int dim = 0)
		: dim_(dim),
		// pseudo_parallel_(true)
		pseudo_parallel_(false)
		{}

		void print() const
		{
			switch(dim_) {
				case 2: {
					data2_.mesh.describe(std::cout);
					// print_boundary_info(data2_.mesh, false);
					break;
				}

				case 3: {
					data3_.mesh.describe(std::cout);
					break;
				}

				case 4: {
					data4_.mesh.describe(std::cout);
					break;
				}

				default: {
					assert(false);
					break;
				}
			}
		}

		void init(Mesh &mesh)
		{
			dim_  = mesh.Dimension();
			geom_ = mesh.GetElement(0)->GetType();
			bdr_geom_ = (mesh.GetNBE() == 0)? 0 : mesh.GetBdrElement(0)->GetType();

			element_to_element_map_.resize(mesh.GetNE());

			for(std::size_t i = 0; i < element_to_element_map_.size(); ++i) {
				element_to_element_map_[i] = i;
			}

			switch(dim_) {
				case 2: {
					convert(mesh, data2_.mesh);
					break;
				}

				case 3: {
					convert(mesh, data3_.mesh);
					break;
				}

				case 4: {
					convert(mesh, data4_.mesh);
					break;
				}

				default: {
					assert(false);
					break;
				}
			}
		}


		template<Integer Dim>
		void refine(const std::vector<Integer> &elems, Data<Dim> &data)
		{
			// using ES = mars::GlobalNewestVertexEdgeSelect<decltype(data.mesh)>;
			// using ES = mars::OldestEdgeSelect<decltype(data.mesh)>;
			using ES = mars::GloballyUniqueLongestEdgeSelect<decltype(data.mesh)>;

			auto &b = data.bisection;
			auto &mesh = data.mesh;

			if(pseudo_parallel_) {
				if(!data.is_initialized) {
					Map map(0, 1);
					
					data.map = std::make_shared<Map>(0, 1);
					data.map->resize(mesh.n_nodes(), 0);
					data.map->identity();

					auto edge_select = std::make_shared<ES>(*data.map);
					b.set_edge_select(edge_select);
					b.edge_select()->update(mesh);
				}

				auto &map = *data.map;

				
				b.tracking_begin();

				b.refine(elems);

				Integer max_iter = 20;
				for(Integer i = 0; i < max_iter; ++i) {
					std::cout << "iter: " << (i + 1) << "/" << max_iter << std::endl;
					
					map.resize(mesh.n_nodes(), 0);
					map.identity();

					b.set_fail_if_not_refine(i == max_iter -1);

					b.edge_select()->update(mesh);

					if(b.refine_incomplete()) {
						break;
					}
				}

				b.tracking_end();

				if(!mesh.is_conforming()) {
					b.undo(); 
				}

			} else {
				b.refine(elems);
			}

			if(!mesh.is_conforming()) {
				std::cerr << "[Warning] encountered non-conforming mesh undoing refinement" << std::endl;
				assert(false);
			}
		}

		void refine(
			const Array<Refinement> &marked_elements,
			Mesh &mesh)
		{

			std::vector<Integer> elems(marked_elements.Size());

			// std::cout << "marked: ";
			for(int i = 0; i < marked_elements.Size(); ++i) {
				// std::cout << marked_elements[i].index << " ";
				elems[i] = element_to_element_map_[marked_elements[i].index];
			}

			// std::cout << std::endl;

			switch(dim_) {
				case 2: {
					refine(elems, data2_);
					convert(data2_.mesh, mesh);
					break;
				}

				case 3: {
					refine(elems, data3_);
					convert(data3_.mesh, mesh);
					break;
				}

				case 4: {
					refine(elems, data4_);
					convert(data4_.mesh, mesh);
					break;
				}

				default: {
					assert(false);
					break;
				}
			}

            // mesh.Print();
            mesh.last_operation = Mesh::REFINE;
            mesh.sequence++;
        }

		template<Integer Dim>
		static void convert(Mesh &in, mars::Mesh<Dim, Dim> &out)
		{
			//TODO handle element attribute

			using Point = mars::Vector<mars::Real, Dim>;

			Array<int> vertices;

			out.clear();

			out.reserve(
				in.GetNE(),
				in.GetNV()
			);

			for(int i = 0; i < in.GetNE(); ++i) {

				Elem<Dim> elem;

				auto &e = *in.GetElement(i);

				assert(e.GetNVertices() == Dim+1);

				e.GetVertices(vertices);
				
				for(int k = 0; k < e.GetNVertices(); ++k) {
					elem.nodes[k] = vertices[k];
				}

				out.add_elem(elem);	
			}

			for(int i = 0; i < in.GetNV(); ++i) {
				Point p;

				auto in_p = in.GetVertex(i);

				for(int d = 0; d < Dim; ++d) {
					p(d) = in_p[d];
				}

				out.add_point(p);
			}

			//can copy it from mfem
			out.update_dual_graph();

			std::map<mars::Side<Dim>, Integer> b_to_index;
			mars::Side<Dim> temp;

			for(int i = 0; i < in.GetNBE(); ++i) {
				Point p;

				auto &be = *in.GetBdrElement(i);
				be.GetVertices(vertices);


				for(int k = 0; k < be.GetNVertices(); ++k) {
					temp[k] = vertices[k];
				}

				temp.fix_ordering();
				b_to_index[temp] = i;
			}

			mars::Simplex<Dim, Dim-1> side;
			for(Integer i = 0; i < out.n_elements(); ++i) {
				if(!out.is_active(i) || !out.is_boundary(i)) continue;

				auto &e = out.elem(i);

				for(Integer k = 0; k < n_sides(e); ++k) {
					e.side(k, side);

					temp = mars::Side<Dim>(side.nodes);

					auto it = b_to_index.find(temp);
					if(it == b_to_index.end()) continue;

					auto bdr_index = it->second;
					auto attr = in.GetBdrAttribute(bdr_index); 

					e.side_tags[k] = attr;
				}
			}
		}

		template<Integer Dim>
		void convert(mars::Mesh<Dim, Dim> &in, Mesh &out)
		{
			using Point = mars::Vector<mars::Real, Dim>;

			const auto n_active_elements = in.n_active_elements();
			assert(n_active_elements > 0);
			
			// CoarseFineTransformations cft
			Mesh temp(Dim,
					  in.n_nodes(),
					  n_active_elements,
					  in.n_boundary_sides()
					);

			// out = temp;

			for(Integer i = 0; i < in.n_nodes(); ++i) {
				temp.AddVertex(&in.point(i)[0]);
			}

			element_to_element_map_.resize(n_active_elements);

			mars::Simplex<Dim, Dim-1> side;
			Integer el_index = 0;
			for(Integer i = 0; i < in.n_elements(); ++i) {
				if(!in.is_active(i)) continue;

				element_to_element_map_[el_index++] = i;

				auto * e = temp.NewElement(geom_);

				std::vector<int> vertices;
				vertices.insert(vertices.begin(), in.elem(i).nodes.begin(), in.elem(i).nodes.end());
				e->SetVertices(&vertices[0]);

				e->SetAttribute(1);

				temp.AddElement(e);

				if(in.is_boundary(i)) {
					const auto &me = in.elem(i);
					for(Integer k = 0; k < n_sides(me); ++k) {
						if(!in.is_boundary(i, k)) continue;
						if(me.side_tags[k] < 0) continue;

						me.side(k, side);

						auto * be = temp.NewElement(bdr_geom_);

						vertices.insert(vertices.begin(), side.nodes.begin(), side.nodes.end());
						be->SetVertices(&vertices[0]);
						be->SetAttribute(me.side_tags[k]);
						temp.AddBdrElement(be);
					}
				}
			}

			if(Dim == 4) {
				temp.PrepareFinalize4D();
				temp.FinalizeTopology();
				temp.Finalize();
			} else {
				temp.FinalizeTopology();
				temp.Finalize();
			}

			out = temp;
		}

	private:
		int dim_;
		int geom_;
		int bdr_geom_;

		Data<2> data2_;
		Data<3> data3_;
		Data<4> data4_;

		std::vector<Integer> element_to_element_map_;
		bool pseudo_parallel_;
		//use this for global dofs
		// Map mesh_map_;
	};


	double NDLSRefiner::GetNorm(const Vector &local_err, Mesh &mesh) const
	{
	#ifdef MFEM_USE_MPI
	    ParMesh *pmesh = dynamic_cast<ParMesh*>(&mesh);
	    if (pmesh)
	    {
	        return ParNormlp(local_err, total_norm_p, pmesh->GetComm());
	    }
	#endif
	    return local_err.Normlp(total_norm_p);
	}
	 
	    
	NDLSRefiner::NDLSRefiner(ErrorEstimator &est)
	    : estimator(est)
	{
	        aniso_estimator = dynamic_cast<AnisotropicErrorEstimator*>(&estimator);
	        total_norm_p = std::numeric_limits<double>::infinity();
	        total_err_goal = 0.0;
	        total_fraction = 0.5;
	        BetaA = 0.001;
	        BetaMode = 1;
	        refinementstrategy = 0; 
	        max_elements = std::numeric_limits<long>::max();
	        total_error_estimate = -1;
	        child_faces.SetSize(0);
	        prev_num_elements = 0;
	        prev_num_vertices = 0;
	        marked_faces.SetSize(0);
	        num_marked_faces = 0L;
	        current_sequence = -1;
	        version_difference =false; 
	        
	        non_conforming = -1;
	        nc_limit = 0;

	        impl = nullptr;
	}

	NDLSRefiner::~NDLSRefiner()
	{
		delete impl;
	}
	    
	int NDLSRefiner::ApplyImpl(Mesh &mesh)
	{
		if(!impl) {
			impl = new RefinerImpl();
			impl->init(mesh);
		}

		num_marked_faces = 0;

		current_sequence = mesh.GetSequence();

		const long num_elements = mesh.GetGlobalNE();
		if (num_elements >= max_elements) { return STOP; }

		const int NE = mesh.GetNE();
		const Vector &local_err = estimator.GetLocalErrors();
		MFEM_ASSERT(local_err.Size() == NE, "invalid size of local_err");

		double total_err = GetNorm(local_err, mesh);
		if (total_err <= total_err_goal) { return STOP; }


//refinement selection algorithm

			EtaCalc(mesh,local_err);

		sort(eta.rbegin(),eta.rend());

    //eta is now sorted in descending order by eta.first . Now we find the minimal subset as defined in MinimalSubset_Sum.

		long refinementindex = MinimalSubset_Sum(eta,total_fraction,1);

    // we refine according to the chosen refinement strategy

		switch (refinementstrategy)
		{
            //refine both
			case 1 :
			{
				Table *face_element = mesh.GetFaceToElementTable();
				Array<int> row;
				Array<long> elements;
				for (int el = 0; el <= refinementindex; el++)
				{
					face_element->GetRow(eta[el].second,row);
					if (row.Size() ==2){
						elements.Append(row[0]);
						elements.Append(row[1]);

					} else{
                    //edge face
						elements.Append(row[0]);

					}
				}
            //remove duplicates.
				elements.Sort();
				elements.Unique();
				marked_elements.SetSize(0);
            //mark for refinement
				for(int el = 0; el<elements.Size(); ++el){
					marked_elements.Append(Refinement(elements[el]));
				}

				if (aniso_estimator)
				{
					const Array<int> &aniso_flags = aniso_estimator->GetAnisotropicFlags();
					if (aniso_flags.Size() > 0)
					{
						for (int i = 0; i < marked_elements.Size(); i++)
						{
							Refinement &ref = marked_elements[i];
							ref.ref_type = aniso_flags[ref.index];
						}
					}
				}
				num_marked_elements = mesh.ReduceInt(marked_elements.Size());
				if (num_marked_elements == 0) { return STOP; }

				// mesh.GeneralRefinement(marked_elements, non_conforming, nc_limit);
				impl->refine(marked_elements, mesh);

				// {
				// 	std::ofstream os("mymesh.MFEM");
				// 	mesh.Print(os);
				// 	os.close();
				// }

				// assert(false);
				break;
			}

			default: {
				MFEM_ASSERT(false, "strategy not supported");
				break;
			}
		}

		return CONTINUE + REFINED;
	}

	double NDLSRefiner::BetaCalc(int index,  const Mesh &mesh){
		double beta;

		switch (BetaMode)
		{
			case 0 :
			{
				beta = BetaA;
				break;
			}
			case 1 :
			{
				double a = mesh.GetFaceArea(index);
				beta = 1.0/a;
				break;
			}
			default :
			mfem_error("NDLSRefiner::BetaCalc : Not applicable BetaMode");
			break;
		}
		return beta;
	}


	void NDLSRefiner::EtaCalc(Mesh &mesh, const Vector &local_err)
	{
		int mfsize = child_faces.Size();
		int iter = 0;

		Table *face_element= mesh.GetFaceToElementTable();
		eta.clear();
		eta.reserve(face_element->Size());
		Array<int> row;
		for (int el = 0; el<(face_element->Size()); ++el)
		{
			face_element->GetRow(el,row);
			if(version_difference == true){

				if (row.Size() ==2) {
                //internal face
					double tau1 = local_err[row[0]];
					double tau2 = local_err[row[1]];

					double estimate = (tau1-tau2)*(tau1-tau2)/BetaCalc(el,mesh);
					eta.push_back(std::make_pair(estimate,el));
				}
			} else {
				if (row.Size() ==2) {
                //internal face
					double tau1 = local_err[row[0]];
					double tau2 = local_err[row[1]];

					double estimate = (tau1-tau2)*(tau1-tau2)/BetaCalc(el,mesh)+1.0/2.0*(tau1*tau1+tau2*tau2);
					eta.push_back(std::make_pair(estimate,el));
				} else {
					double tau1 = local_err[row[0]];
					double estimate =tau1*tau1*(1.0/BetaCalc(el,mesh)+.5);
					eta.push_back(std::make_pair(estimate,el));
				}
			}
		}
	}

    //function for computing the minimal subset of set such that
    // sum(subset)>=weight*sum(set). The tolerance dictates the precision of the bisection search being performed. Tolerance of one gives the actual minimum set, tolerance of n gives the nth bisection iteration away from the actual minimum set.
	long NDLSRefiner::MinimalSubset_Sum(const std::vector< std::pair <double,int> > &set, double weight, unsigned int tolerance)
	{
		long bisectpoint = std::floor(weight*set.size());
    //initial sum and check
		double leftside = 0.0;
		double rightside = 0.0;
		unsigned long leftindex= 0;
		unsigned long rightindex =set.size()-1;
		for(int i = 0; i<set.size(); ++i)
		{
			rightside+=set[i].first;
			if (i<bisectpoint){
				leftside+=set[i].first;
			}
		}
		bool moveleft=(leftside>=weight*rightside);
		if (moveleft)
		{
			rightindex=bisectpoint;
		} else {
			leftindex=bisectpoint;
		}
		if ((rightindex-leftindex)<=tolerance){
			return bisectpoint;
		}
    //now we iterate while tolerance is not achieved
		while((rightindex-leftindex)>tolerance){
        //bisect
			bisectpoint = (rightindex-leftindex)/2+leftindex;
			if(moveleft){
            //then previous bisectpoint is now rightindex and rightside grows and leftside shrinks.
				double holder = 0.0;
				for(long i = rightindex-1; i>=bisectpoint; --i) {
					holder +=set[i].first;;
				}
				leftside -=holder;
			} else {
            //then previous bisectpoint is now leftindex and rightside shrinks and leftside grows.
				double holder = 0.0;
				for(long i = leftindex; i<bisectpoint; ++i) {
					holder +=set[i].first;
				}
				leftside +=holder;;
			}
			moveleft=(leftside>=weight*rightside);
			if (moveleft)
			{
				rightindex=bisectpoint;
			} else {
				leftindex=bisectpoint;
			}
		}
		return bisectpoint; 
	}

	double NDLSRefiner::GetTotalErrorEstimate(Mesh &mesh) {

        //recalculate local element errors
		const int NE = mesh.GetNE();
		const Vector &local_err = estimator.GetLocalErrors();
		MFEM_ASSERT(local_err.Size() == NE, "invalid size of local_err");
        //calculate face errors
		EtaCalc(mesh,local_err);
        //sum these face errors
		double total_face_error = 0.0;
		for (int i = 0; i<eta.size(); ++i){
			total_face_error+=eta[i].first;
		}
		total_face_error = std::sqrt(total_face_error);
		return total_face_error;
	}

	void NDLSRefiner::Reset()
	{
		estimator.Reset();
		current_sequence = -1;
		num_marked_faces = 0;
	}

}
