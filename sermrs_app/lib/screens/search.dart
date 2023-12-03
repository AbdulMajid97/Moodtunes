import 'package:flutter/material.dart';
import 'package:sermrs_app/screens/widgets/data.dart';

class Search extends StatefulWidget {
  const Search({super.key});

  @override
  State<Search> createState() => _SearchState();
}

class _SearchState extends State<Search> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color.fromRGBO(18, 18, 18, 80),
      body: Container(
        padding: const EdgeInsets.fromLTRB(10, 5, 10, 10),
        child: ListView(
          children: const [
            Text(
              'Search',
              style: TextStyle(
                color: Colors.white,
                fontSize: 34,
                fontFamily: 'LibreFranklin',
                fontWeight: FontWeight.bold,
              ),
            ),
            SearchWidget(),
            TopGenre(),
          ],
        ),
      ),
    );
  }
}

class SearchWidget extends StatelessWidget {
  const SearchWidget({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      height: 60,
      margin: const EdgeInsets.symmetric(
        horizontal: 0,
        vertical: 30,
      ),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(5),
      ),
      child: Row(
        children: const [
          SizedBox(width: 10),
          Icon(
            Icons.search_sharp,
            size: 30,
            color: Color.fromRGBO(83, 83, 83, 1),
          ),
          SizedBox(width: 10),
          Text(
            'Artists or Songs,',
            style: TextStyle(
              fontFamily: 'LibreFranklin',
              color: Color.fromRGBO(83, 83, 83, 1),
              fontSize: 17,
            ),
            softWrap: true,
            maxLines: 1,
            overflow: TextOverflow.fade,
          ),
        ],
      ),
    );
  }
}

class TopGenre extends StatelessWidget {
  const TopGenre({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          'You top genres',
          style: TextStyle(
            color: Colors.white,
            fontFamily: 'LibreFranklin',
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 10),
        Tiles(Data().genres),
        const SizedBox(height: 30),
        const Text(
          'Browse all',
          style: TextStyle(
            fontFamily: 'LibreFranklin',
            color: Colors.white,
            fontSize: 22,
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(height: 10),
        Tiles(Data().browseall),
      ],
    );
  }
}

class Tiles extends StatelessWidget {
  List<String> something;
  Tiles(this.something, {super.key});
  @override
  Widget build(BuildContext context) {
    return GridView(
      shrinkWrap: true,
      physics: const ScrollPhysics(),
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        childAspectRatio: 16 / 9,
        crossAxisSpacing: 20,
        mainAxisSpacing: 20,
      ),
      children: something.map((imageUrl) {
        return Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(5),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(5),
              child: Image.asset(
                imageUrl,
                fit: BoxFit.cover,
              ),
            ));
      }).toList(),
    );
  }
}
