import 'package:flutter/material.dart';
import 'package:getwidget/getwidget.dart';
import 'package:sermrs_app/screens/widgets/data.dart';

class Library extends StatefulWidget {
  const Library({super.key});

  @override
  State<Library> createState() => _LibraryState();
}

class _LibraryState extends State<Library> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color.fromRGBO(18, 18, 18, 1),
      body: ListView(
        children: [
          const Header(),
          Row(
            children: const [
              SizedBox(
                width: 10,
              ),
              RoundedCards('Playlists'),
              SizedBox(
                width: 10,
              ),
              RoundedCards('Artists'),
            ],
          ),
          const SizedBox(
            height: 25,
          ),
          Padding(
            padding: const EdgeInsets.all(10.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: const [
                    Icon(
                      Icons.compare_arrows_rounded,
                      color: Colors.white,
                      size: 15,
                    ),
                    SizedBox(width: 10),
                    Text(
                      'Recently played',
                      style: TextStyle(
                        color: Colors.white,
                        fontFamily: 'LibreFranklin',
                        fontWeight: FontWeight.w100,
                        fontSize: 15,
                      ),
                    ),
                  ],
                ),
                const Icon(
                  Icons.add_box_outlined,
                  color: Colors.white,
                  size: 20,
                ),
              ],
            ),
          ),
          ...Data().library.map((val) {
            return GFListTile(
              avatar: GFAvatar(
                backgroundImage: AssetImage(
                  val['image'].toString(),
                ),
                radius: 30,
                shape: val['shape'] as dynamic,
              ),
              title: Text(
                val['name'].toString(),
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                  fontSize: 18,
                ),
              ),
              subTitle: Text(
                val['subtitle'].toString(),
                style: const TextStyle(
                  color: Color.fromRGBO(167, 167, 167, 1),
                  fontSize: 14,
                ),
              ),
            );
          }).toList(),
          Tiles('Add artists', GFAvatarShape.circle),
        ],
      ),
    );
  }
}

class Header extends StatelessWidget {
  const Header({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 20),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: const [
              CircleAvatar(
                backgroundColor: Colors.deepOrange,
                radius: 16,
                child: Text(
                  'S',
                  style: TextStyle(
                    fontSize: 25,
                    color: Colors.black,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
              SizedBox(width: 10),
              Text(
                'Your Library',
                style: TextStyle(
                  fontFamily: 'LibreFranklin',
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ],
          ),
          Row(
            children: const [
              Icon(
                Icons.search,
                color: Colors.white,
                size: 30,
              ),
              SizedBox(
                width: 15,
              ),
              Icon(
                Icons.add,
                color: Colors.white,
                size: 30,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class RoundedCards extends StatelessWidget {
  final String text;
  const RoundedCards(this.text, {super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: Colors.white, width: 1),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(
          horizontal: 8.0,
          vertical: 4,
        ),
        child: Text(
          text,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 18,
          ),
        ),
      ),
    );
  }
}

class Tiles extends StatelessWidget {
  String title;
  GFAvatarShape shape;
  Tiles(this.title, this.shape, {super.key});

  @override
  Widget build(BuildContext context) {
    return GFListTile(
      avatar: GFAvatar(
        radius: 30,
        backgroundColor: Colors.grey[900],
        shape: shape,
        child: const Icon(
          Icons.add,
          size: 40,
          color: Colors.white54,
        ),
      ),
      title: Text(
        title,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 18,
        ),
      ),
    );
  }
}
